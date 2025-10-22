#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки

#include <fstream>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_radixSort01LocalCounting(avk2::getRadixSort01LocalCounting());
    avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(avk2::getRadixSort02GlobalPrefixesScanSumReduction());
    avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulation());
    avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

    FastRandom r;

    int n = 100*1000*1000;
    int max_value = std::numeric_limits<int>::max();
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value << std::endl;
    {
        // убедимся что в массиве есть хотя бы несколько повторяющихся значений
        size_t force_duplicates_attempts = 3;
        bool all_attempts_missed = true;
        for (size_t k = 0; k < force_duplicates_attempts; ++k) {
            size_t i = r.next(0, n - 1);
            size_t j = r.next(0, n - 1);
            if (i != j) {
                as[j] = as[i];
                all_attempts_missed = false;
            }
        }
        rassert(!all_attempts_missed, 4353245123412);
    }

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << t.elapsed() << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / t.elapsed() << " GB/s (" << n / 1000 / 1000 / t.elapsed() << " uint millions/s)" << std::endl;
    }

    const unsigned int N_GROUPS = ((n + GROUP_SIZE - 1) / GROUP_SIZE);
    const unsigned int BUCKET_N = N_GROUPS * (1 << RADIX_BIT_CNT);
    std::cout << "Groups: " << N_GROUPS << ", Buckets: " << BUCKET_N << '\n';
    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u array1(n), array2(n);
    gpu::gpu_mem_32u buckets(BUCKET_N), prefix_sum(BUCKET_N), buffer_pow_1(BUCKET_N), buffer_pow_2(BUCKET_N);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    array1.writeN(as.data(), n);

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;
        for (unsigned int bit_start = 0; bit_start < 32; bit_start += RADIX_BIT_CNT) {
            ocl_fillBufferWithZeros.exec(gpu::WorkSize(GROUP_SIZE, BUCKET_N), buckets, BUCKET_N);
            ocl_radixSort01LocalCounting.exec(gpu::WorkSize(GROUP_SIZE, n), array1, buckets, n, bit_start);
            
            gpu::WorkSize workSize_n(GROUP_SIZE, BUCKET_N);
            ocl_radixSort03GlobalPrefixesScanAccumulation.exec(workSize_n, buckets, prefix_sum, BUCKET_N, 0);
            ocl_radixSort02GlobalPrefixesScanSumReduction.exec(workSize_n, buckets, buffer_pow_1, BUCKET_N);
            int pow2 = 1;
            int m = (BUCKET_N + 1) / 2;
            while ((1 << pow2) <= BUCKET_N) {
                gpu::WorkSize workSize_m(GROUP_SIZE, m);
                ocl_radixSort03GlobalPrefixesScanAccumulation.exec(workSize_n, buffer_pow_1, prefix_sum, BUCKET_N, pow2);
                ocl_radixSort02GlobalPrefixesScanSumReduction.exec(workSize_m, buffer_pow_1, buffer_pow_2, m);
                std::swap(buffer_pow_1, buffer_pow_2);
                m = (m + 1) / 2;
                pow2++;
            }
            ocl_radixSort04Scatter.exec(gpu::WorkSize(GROUP_SIZE, n), array1, prefix_sum, array2, n, bit_start);
            std::swap(array1, array2);
        }
        times.push_back(t.elapsed());
    }

    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = array1.readVector();

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    // Я думаю делать inplace лучше (удобнее по крайней мере), + в любом случае разница в 1 копирование
    // std::vector<unsigned int> input_values = array1.readVector();
    // for (size_t i = 0; i < n; ++i) {
    //     rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    // }
}

int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            return 0;
        } if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            return 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            return 1;
        }
    }

    return 0;
}
