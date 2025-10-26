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
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

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
    int num_active_bit = sizeof(max_value) * 8 - __builtin_clz(max_value);
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
    gpu::gpu_mem_32u copy_input_gpu(n);
    gpu::gpu_mem_32u buffer_output_gpu(n);

    unsigned int num_batches = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    unsigned int size_vec_for_batches = num_batches * BITS_COUNT;
    gpu::gpu_mem_32u batches_counter(size_vec_for_batches);
    gpu::gpu_mem_32u buffer1_pow2_sum_gpu(size_vec_for_batches);
    gpu::gpu_mem_32u buffer2_pow2_sum_gpu(size_vec_for_batches);
    gpu::gpu_mem_32u prefix_sum_accum_gpu(size_vec_for_batches);


    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    input_gpu.copyTo(copy_input_gpu, n * sizeof(unsigned int));

    gpu::WorkSize workSize_n(GROUP_SIZE, n);
    gpu::WorkSize workSize_batches(GROUP_SIZE, size_vec_for_batches);
    ocl_fillBufferWithZeros.exec(workSize_batches, prefix_sum_accum_gpu, size_vec_for_batches);
    ocl_fillBufferWithZeros.exec(workSize_batches, buffer1_pow2_sum_gpu, size_vec_for_batches);
    ocl_fillBufferWithZeros.exec(workSize_batches, buffer2_pow2_sum_gpu, size_vec_for_batches);
    ocl_fillBufferWithZeros.exec(workSize_n, buffer_output_gpu, n);
    ocl_fillBufferWithZeros.exec(workSize_batches, batches_counter, size_vec_for_batches);


    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 1; ++iter) {
        timer t;
        for (unsigned int bit_start = 0; bit_start < num_active_bit; bit_start += BITS_BATCH) {

            if constexpr (DEBUG == 1) {
                std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

                std::vector<unsigned int> input_values = copy_input_gpu.readVector();
                std::cout << "a_gpu:";
                for (size_t i = 0; i < n; ++i) std::cout << " " << input_values[i];
                std::cout << std::endl;

                std::cout << "a_gpu bits: ";
                for (int i = 0; i < n; ++i) {
                    for (int j = num_active_bit - 1; j >= 0; --j) {
                        if (j == bit_start + BITS_BATCH - 1) std::cout << "[";
                        std::cout << ((input_values[i] >> j) & 1);
                        if (j == bit_start) std::cout << "]";
                    }
                    std::cout << " ";
                }
                std::cout << std::endl;
            }


            ocl_fillBufferWithZeros.exec(workSize_batches, batches_counter, size_vec_for_batches);
            ocl_radixSort01LocalCounting.exec(workSize_n, copy_input_gpu, batches_counter, n, bit_start);

            if constexpr (DEBUG == 1) {
                std::vector<unsigned int> batches_values = batches_counter.readVector();
                std::cout << "batches_values:";
                for (size_t i = 0; i < size_vec_for_batches; ++i) std::cout << " " << batches_values[i];
                std::cout << std::endl;
            }

            ocl_fillBufferWithZeros.exec(workSize_batches, prefix_sum_accum_gpu, size_vec_for_batches);

            unsigned int pow2 = 0;
            ocl_radixSort03GlobalPrefixesScanAccumulation.exec(workSize_batches, batches_counter, prefix_sum_accum_gpu, size_vec_for_batches, pow2);

            batches_counter.copyTo(buffer1_pow2_sum_gpu, size_vec_for_batches * sizeof(unsigned int));
            unsigned int copy_n = size_vec_for_batches;
            while (copy_n > 1) {
                ocl_radixSort02GlobalPrefixesScanSumReduction.exec(workSize_batches, buffer1_pow2_sum_gpu, buffer2_pow2_sum_gpu, copy_n);
                ocl_radixSort03GlobalPrefixesScanAccumulation.exec(workSize_batches, buffer2_pow2_sum_gpu, prefix_sum_accum_gpu, size_vec_for_batches, ++pow2);
                std::swap(buffer1_pow2_sum_gpu, buffer2_pow2_sum_gpu);
                copy_n = div_ceil(copy_n, (unsigned int)2);
            }

            if constexpr (DEBUG == 1) {
                std::vector<unsigned int> prefix_sum_accum_gpu_values = prefix_sum_accum_gpu.readVector();
                std::cout << "prefix_values:";
                for (size_t i = 0; i < size_vec_for_batches; ++i) std::cout << " " << prefix_sum_accum_gpu_values[i];
                std::cout << std::endl;
            }

            ocl_radixSort04Scatter.exec(workSize_n, copy_input_gpu, prefix_sum_accum_gpu, buffer_output_gpu, n, bit_start);

            if constexpr (DEBUG == 1) {
                std::vector<unsigned int> output_values = buffer_output_gpu.readVector();
                std::cout << "b_gpu:";
                for (size_t i = 0; i < n; ++i) std::cout << " " << output_values[i];
                std::cout << std::endl;

                std::cout << "b_gpu bits: ";
                for (int i = 0; i < n; ++i) {
                    for (int j = num_active_bit - 1; j >= 0; --j) {
                        if (j == bit_start + BITS_BATCH - 1) std::cout << "[";
                        std::cout << ((output_values[i] >> j) & 1);
                        if (j == bit_start) std::cout << "]";
                    }
                    std::cout << " ";
                }
                std::cout << std::endl;
                std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
            }

            std::swap(copy_input_gpu, buffer_output_gpu);
        }

        std::swap(copy_input_gpu, buffer_output_gpu);
        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = buffer_output_gpu.readVector();

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
