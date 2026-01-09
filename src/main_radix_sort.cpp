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

void map_scan(
    uint n,
    gpu::gpu_mem_32u& input_gpu, // input
    gpu::gpu_mem_32u& buffer1_pow2_sum_gpu, // tmp
    gpu::gpu_mem_32u& buffer2_pow2_sum_gpu, // tmp
    gpu::gpu_mem_32u& prefix_sum_accum_gpu, // outputs,
    uint bits,
    uint digit,
    uint clazz,
    ocl::KernelSource& ocl_map,
    ocl::KernelSource& ocl_sum_reduction,
    ocl::KernelSource& ocl_prefix_accumulation,
    ocl::KernelSource& ocl_fillBufferWithZeros)
{
    // map
    ocl_map.exec(gpu::WorkSize(GROUP_SIZE, 1, n, 1), n, input_gpu, buffer1_pow2_sum_gpu, bits, digit, clazz);
    // std::cout << "buf1 ";
    // for (auto&& i : buffer1_pow2_sum_gpu.readVector()) {
    //     std::cout << i << " ";
    // }
    // std::cout << "\n";
    ocl_fillBufferWithZeros.exec(gpu::WorkSize(GROUP_SIZE, 1, n, 1), buffer2_pow2_sum_gpu, n);
    ocl_map.exec(gpu::WorkSize(GROUP_SIZE, 1, n, 1), n, input_gpu, prefix_sum_accum_gpu, bits, digit, clazz);

    // scan
    gpu::gpu_mem_32u* buf1_ptr = std::addressof(buffer1_pow2_sum_gpu);
    gpu::gpu_mem_32u* buf2_ptr = std::addressof(buffer2_pow2_sum_gpu);
    ocl_prefix_accumulation.exec(gpu::WorkSize(GROUP_SIZE, 1, n, 1), *buf1_ptr, prefix_sum_accum_gpu, n, 0);

    for (unsigned int k = 0; k < floor(log2(n)); k++) {
        unsigned int reduction_size = n / (1 << (k + 1));
        ocl_sum_reduction.exec(gpu::WorkSize(GROUP_SIZE, 1, reduction_size, 1), *buf1_ptr, *buf2_ptr, reduction_size);
        ocl_prefix_accumulation.exec(gpu::WorkSize(GROUP_SIZE, 1, n, 1), *buf2_ptr, prefix_sum_accum_gpu, n, k + 1);
        std::swap(buf1_ptr, buf2_ptr);
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
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_fillBufferWithZeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01GlobalPrefixesScanSumReduction(ocl::getRadixSort01GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanAccumulation(ocl::getRadixSort02GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort03Scatter(ocl::getRadixSort03Scatter());
    ocl::KernelSource ocl_radixSort04Map(ocl::getRadixSort04Map());

    avk2::KernelSource vk_fillBufferWithZeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_radixSort01LocalCounting(avk2::getRadixSort01LocalCounting());
    avk2::KernelSource vk_radixSort02GlobalPrefixesScanSumReduction(avk2::getRadixSort02GlobalPrefixesScanSumReduction());
    avk2::KernelSource vk_radixSort03GlobalPrefixesScanAccumulation(avk2::getRadixSort03GlobalPrefixesScanAccumulation());
    avk2::KernelSource vk_radixSort04Scatter(avk2::getRadixSort04Scatter());

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
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u buffer1_pow2_sum_gpu(n), buffer2_pow2_sum_gpu(n);
    gpu::gpu_mem_32u buffer1_scatter_gpu(n), buffer2_scatter_gpu(n);
    gpu::gpu_mem_32u buffer_output_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // for (auto&& i : as) {
    //     std::cout << i << " ";
    // }
    // std::cout << "\n";

    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    buffer1_pow2_sum_gpu.fill(255);
    buffer2_pow2_sum_gpu.fill(255);
    input_gpu.copyToN(buffer1_scatter_gpu, n);
    buffer2_scatter_gpu.fill(255);
    buffer_output_gpu.fill(255);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {
            const uint bits = 2;

            for (uint digit = 0; digit < 32 / bits; digit++) {
                uint offset = 0;
                for (uint clazz = 0; clazz < (1 << bits); clazz++) {
                    map_scan(n,
                        buffer1_scatter_gpu,
                        buffer1_pow2_sum_gpu,
                        buffer2_pow2_sum_gpu,
                        buffer_output_gpu, // pref_sum
                        bits,
                        digit,
                        clazz,
                        ocl_radixSort04Map,
                        ocl_radixSort01GlobalPrefixesScanSumReduction,
                        ocl_radixSort02GlobalPrefixesScanAccumulation,
                        ocl_fillBufferWithZeros);

                    // for (auto&& i : debug::prettyBits(buffer1_scatter_gpu.readVector(), max_value, digit * bits, bits)) {
                    //     std::cout << i << " ";
                    // }
                    // std::cout << "\n";
                    // for (auto&& i : buffer_output_gpu.readVector()) {
                    //     std::cout << i << " ";
                    // }
                    // std::cout << "\n";

                    // printf("bits=%d digit=%d offset=%d clazz=%s\n", bits, digit, offset, debug::prettyBits({ clazz }, 1 << bits - 1, 0, bits)[0].c_str());

                    ocl_radixSort03Scatter.exec(gpu::WorkSize(GROUP_SIZE, 1, n, 1),
                        n,
                        buffer1_scatter_gpu,
                        buffer2_scatter_gpu,
                        buffer_output_gpu,
                        bits, digit, offset, clazz);

                    uint _offset;
                    buffer_output_gpu.readN(&_offset, 1, n - 1); // TODO: check
                    // std::cout << offset << " + " << _offset << "\n";
                    offset += _offset;

                    // std::cout << "\n";
                }

                std::swap(buffer1_scatter_gpu, buffer2_scatter_gpu); // TODO: ptr swap ???
            }

        } else if (context.type() == gpu::Context::TypeCUDA) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // cuda::fill_buffer_with_zeros();
            // cuda::radix_sort_01_local_counting();
            // cuda::radix_sort_02_global_prefixes_scan_sum_reduction();
            // cuda::radix_sort_03_global_prefixes_scan_accumulation();
            // cuda::radix_sort_04_scatter();
        } else if (context.type() == gpu::Context::TypeVulkan) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // vk_fillBufferWithZeros.exec();
            // vk_radixSort01LocalCounting.exec();
            // vk_radixSort02GlobalPrefixesScanSumReduction.exec();
            // vk_radixSort03GlobalPrefixesScanAccumulation.exec();
            // vk_radixSort04Scatter.exec();
        } else {
            rassert(false, 4531412341, context.type());
        }

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = buffer1_scatter_gpu.readVector();

    // std::cout << "sorted ";
    // for (auto&& i : sorted) {
    //     std::cout << i << " ";
    // }
    // std::cout << "\n";
    // std::cout << "gputed ";
    // for (auto&& i : gpu_sorted) {
    //     std::cout << i << " ";
    // }
    // std::cout << "\n";

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
