#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/fast_random.h>
#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки

#include <fstream>

// #define _MY_DEBUG

void debug_array(
    const std::vector<unsigned int>& arr,
    const std::string& label,
    unsigned int max_value,
    unsigned int offset)
{
#ifdef _MY_DEBUG
    std::cout << label << " (offset: " << offset << ")" << std::endl;
    auto ss = debug::prettyBits(arr, max_value, offset, RADIX_BITS);
    for (int i = 0; i < arr.size(); i++) {
        if (i % GROUP_SIZE == 0)
            std::cout << "{ ";
        std::cout << ss[i] << " ";
        if ((i + 1) % GROUP_SIZE == 0 || i + 1 == arr.size())
            std::cout << "} ";
    }
    std::cout << std::endl;
#endif
}

void debug_array(
    const gpu::gpu_mem_32u& arr_gpu,
    const std::string& label,
    unsigned int max_value,
    unsigned int offset)
{
    debug_array(arr_gpu.readVector(), label, max_value, offset);
}

void debug_array_2d(const gpu::gpu_mem_32u& arr_gpu, unsigned int prefix_sum_size, const std::string& label)
{
#ifdef _MY_DEBUG
    std::vector<unsigned int> arr = arr_gpu.readVector();

    unsigned int size_y = prefix_sum_size >> RADIX_BITS;

    std::cout << label << ": ";
    for (int i = 0; i < size_y; i++) {
        std::cout << "{ ";
        for (int j = 0; j < (1 << RADIX_BITS); j++)
            std::cout << arr[j * size_y + i] << " ";
        std::cout << "} ";
    }
    std::cout << std::endl;
#endif
}

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    FastRandom r;

    int n = 100 * 1000 * 1000;
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

    // Аллоцируем буферы в VRAM
    const unsigned int prefix_sum_size = div_ceil(n, GROUP_SIZE) << RADIX_BITS;

    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u buffer1_gpu(prefix_sum_size),
        buffer2_gpu(prefix_sum_size), buffer3_gpu(prefix_sum_size);
    gpu::gpu_mem_32u buffer_output_gpu1(n), buffer_output_gpu2(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    buffer1_gpu.fill(255);
    buffer2_gpu.fill(255);
    buffer3_gpu.fill(255);
    buffer_output_gpu1.fill(255);
    buffer_output_gpu2.fill(255);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        unsigned int max_offset = (8 * sizeof(max_value) - __builtin_clz(max_value)) - 1;

        for (unsigned int offset = 0; offset <= max_offset; offset += RADIX_BITS) {

#ifdef _MY_DEBUG
            std::cout << std::endl;
            if (offset == 0)
                debug_array(input_gpu, "before", max_value, offset);
            else if (offset / RADIX_BITS % 2 == 1)
                debug_array(buffer_output_gpu1, "before", max_value, offset);
            else
                debug_array(buffer_output_gpu2, "before", max_value, offset);
#endif
            ocl_fillBufferWithZeros.exec(gpu::WorkSize(GROUP_SIZE, prefix_sum_size), buffer1_gpu, prefix_sum_size);

            if (offset == 0)
                ocl_radixSort01LocalCounting.exec(gpu::WorkSize(GROUP_SIZE, n), input_gpu, buffer1_gpu, n, offset);
            else if (offset / RADIX_BITS % 2 == 1)
                ocl_radixSort01LocalCounting.exec(gpu::WorkSize(GROUP_SIZE, n), buffer_output_gpu1, buffer1_gpu, n, offset);
            else
                ocl_radixSort01LocalCounting.exec(gpu::WorkSize(GROUP_SIZE, n), buffer_output_gpu2, buffer1_gpu, n, offset);

            debug_array_2d(buffer1_gpu, prefix_sum_size, "local counting");

            ocl_fillBufferWithZeros.exec(gpu::WorkSize(GROUP_SIZE, prefix_sum_size), buffer3_gpu, prefix_sum_size);
            ocl_radixSort03GlobalPrefixesScanAccumulation.exec(
                gpu::WorkSize(GROUP_SIZE, prefix_sum_size),
                buffer1_gpu,
                buffer3_gpu,
                prefix_sum_size,
                0);

            for (unsigned int window_size = 1; (1 << window_size) <= prefix_sum_size; window_size++) {
                if (window_size % 2 == 1) {
                    ocl_radixSort02GlobalPrefixesScanSumReduction.exec(
                        gpu::WorkSize(GROUP_SIZE, div_ceil(prefix_sum_size, 1u << window_size)),
                        buffer1_gpu,
                        buffer2_gpu,
                        div_ceil(prefix_sum_size, 1u << (window_size - 1)));
                    ocl_radixSort03GlobalPrefixesScanAccumulation.exec(
                        gpu::WorkSize(GROUP_SIZE, prefix_sum_size),
                        buffer2_gpu,
                        buffer3_gpu,
                        prefix_sum_size,
                        window_size);
                } else {
                    ocl_radixSort02GlobalPrefixesScanSumReduction.exec(
                        gpu::WorkSize(GROUP_SIZE, div_ceil(prefix_sum_size, 1u << window_size)),
                        buffer2_gpu,
                        buffer1_gpu,
                        div_ceil(prefix_sum_size, 1u << (window_size - 1)));
                    ocl_radixSort03GlobalPrefixesScanAccumulation.exec(
                        gpu::WorkSize(GROUP_SIZE, prefix_sum_size),
                        buffer1_gpu,
                        buffer3_gpu,
                        prefix_sum_size,
                        window_size);
                }

                // debug_array_2d(buffer1_gpu, prefix_sum_size, "global offset");
            }

            debug_array_2d(buffer3_gpu, prefix_sum_size, "global offset");

            if (offset == 0)
                ocl_radixSort04Scatter.exec(gpu::WorkSize(GROUP_SIZE, n), input_gpu, buffer3_gpu, buffer_output_gpu1, n, offset);
            else if (offset / RADIX_BITS % 2 == 1)
                ocl_radixSort04Scatter.exec(gpu::WorkSize(GROUP_SIZE, n), buffer_output_gpu1, buffer3_gpu, buffer_output_gpu2, n, offset);
            else
                ocl_radixSort04Scatter.exec(gpu::WorkSize(GROUP_SIZE, n), buffer_output_gpu2, buffer3_gpu, buffer_output_gpu1, n, offset);

            if (offset / RADIX_BITS % 2 == 1)
                debug_array(buffer_output_gpu2, "after", max_value, offset);
            else
                debug_array(buffer_output_gpu1, "after", max_value, offset);
        }

        if (max_offset / RADIX_BITS % 2 == 1)
            buffer_output_gpu1.swap(buffer_output_gpu2);

#ifdef _MY_DEBUG
        std::cout << std::endl;

        debug_array(buffer_output_gpu1, "result", max_value, 0);
        debug_array(sorted, "correct", max_value, 0);
#endif

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = buffer_output_gpu1.readVector();

    // debug_array(gpu_sorted, "result", max_value, 0);

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
    }

    // Проверяем что входные данные остались нетронуты (ведь мы их переиспользуем от итерации к итерации)
    std::vector<unsigned int> input_values = input_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(input_values[i] == as[i], 6573452432, input_values[i], as[i]);
    }
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
        }
        if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            return 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            return 1;
        }
    }

    return 0;
}
