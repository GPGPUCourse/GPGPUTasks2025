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

void prefix_sum(ocl::KernelSource prefix_sum_kernel, const gpu::gpu_mem_32u &mem, unsigned int n)
{
    unsigned int k = 1;
    while (1 << (k - 1) < n) {
        prefix_sum_kernel.exec(gpu::WorkSize(256, n / 2), mem, n, k);
        k++;
    }
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_prefixSum(ocl::getPrefixSum());
    ocl::KernelSource ocl_copy(ocl::getCopy());
    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSortMap(ocl::getRadixSortMap());
    ocl::KernelSource ocl_radixSortScatter(ocl::getRadixSortScatter());

    FastRandom r;

    int n = 100*1000*1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    int max_value = 100; // std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
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
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u map_buffer_gpu(n * (1 << DIGITS_SIZE));
    gpu::gpu_mem_32u scatter_buffer_gpu(n * (1 << DIGITS_SIZE));
    gpu::gpu_mem_32u buffer_output_gpu_a(n);
    gpu::gpu_mem_32u buffer_output_gpu_b(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    buffer_output_gpu_a.fill(255);
    buffer_output_gpu_b.fill(255);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        input_gpu.copyToN(buffer_output_gpu_a, n);
        unsigned int all_digits_size = n * (1 << DIGITS_SIZE);

        for (unsigned int k = 0; k < 32; k += DIGITS_SIZE) {
            // std::cout << "Offset " << k << std::endl;
            // for (unsigned int x : buffer_output_gpu_a.readVector()) {
            //     std::cout << x << ' ';
            // }
            // std::cout << std::endl;
            // for (unsigned int x : buffer_output_gpu_a.readVector()) {
            //     std::cout << ((x >> k) & ((1 << DIGITS_SIZE) - 1)) << ' ';
            // }
            // std::cout << std::endl;
            ocl_fillBufferWithZeros.exec(gpu::WorkSize(GROUP_SIZE, all_digits_size), map_buffer_gpu, all_digits_size);
            ocl_radixSortMap.exec(gpu::WorkSize(GROUP_SIZE, all_digits_size), buffer_output_gpu_a, map_buffer_gpu, n, k);
            // for (unsigned int x : map_buffer_gpu.readVector()) {
            //     std::cout << x << ' ';
            // }
            // std::cout << std::endl;
            ocl_copy.exec(gpu::WorkSize(GROUP_SIZE, all_digits_size), map_buffer_gpu, scatter_buffer_gpu,  all_digits_size);
            prefix_sum(ocl_prefixSum, scatter_buffer_gpu, all_digits_size);
            // for (unsigned int x : scatter_buffer_gpu.readVector()) {
            //     std::cout << x << ' ';
            // }
            // std::cout << std::endl;
            ocl_radixSortScatter.exec(gpu::WorkSize(GROUP_SIZE, all_digits_size), buffer_output_gpu_a, map_buffer_gpu, scatter_buffer_gpu, buffer_output_gpu_b, n);
            // for (unsigned int x : buffer_output_gpu_b.readVector()) {
            //     std::cout << x << ' ';
            // }
            // std::cout << std::endl;
            std::swap(buffer_output_gpu_a, buffer_output_gpu_b);
        }

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = buffer_output_gpu_a.readVector();

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
