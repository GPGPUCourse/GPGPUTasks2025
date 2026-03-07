#include <iostream>
#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки
#include "libbase/math.h"
#include "libgpu/shared_device_buffer.h"

#include <fstream>
#include <sys/types.h>
#include <utility>

void debug_mem(const gpu::gpu_mem_32u& mem, const std::string& log_prefix, bool need_bin = false) {
    std::vector<unsigned int> vec = mem.readVector();

    std::cerr << log_prefix << ":\n";
    for (auto& i : vec) {
        std::cerr << i << "\n";
    }
    std::cerr << std::endl;
    if (need_bin) {
        auto bits = debug::toBits(vec);
        for (auto& i : bits) {
            std::cerr << i << "\n";
        }
        std::cerr << std::endl;
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

    uint n = 100*1000*1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага

    // n = 20; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    // max_value = 1000000000; // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага

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
    gpu::gpu_mem_32u arr1_gpu(n), arr2_gpu(n), input_gpu(n);

    const uint GROUPS_COUNT = (n + GROUP_SIZE - 1) / GROUP_SIZE;
    gpu::gpu_mem_32u local_buckets_gpu(GROUPS_COUNT * BUCKET_SIZE);
    gpu::gpu_mem_32u pref_sum_buf_1(GROUPS_COUNT * BUCKET_SIZE);
    gpu::gpu_mem_32u pref_sum_buf_2(GROUPS_COUNT * BUCKET_SIZE);
    gpu::gpu_mem_32u global_pref_sums(GROUPS_COUNT * BUCKET_SIZE);
    local_buckets_gpu.fill(0);
    pref_sum_buf_1.fill(0);
    pref_sum_buf_2.fill(0);
    global_pref_sums.fill(0);

    // gpu::gpu_mem_32u buffer1_gpu(n), buffer2_gpu(n), buffer3_gpu(n), buffer4_gpu(n); // TODO это просто шаблонка, можете переименовать эти буферы, сделать другого размера/типа, удалить часть, добавить новые

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)

    gpu::gpu_mem_32u* in_arr = &arr1_gpu;
    gpu::gpu_mem_32u* out_arr = &arr2_gpu;
    std::vector<double> times;
    const int NUM_ITERS = 1;
    // Запускаем кернел (несколько раз и с замером времени выполнения)
    for (int iter = 0; iter < NUM_ITERS; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;
        gpu::WorkSize workSize(GROUP_SIZE, n);
        input_gpu.copyToN(arr1_gpu, n);
        in_arr = &arr2_gpu;
        out_arr = &arr1_gpu;

        for (uint offset = 0; offset < 8 * sizeof(uint); offset += BUCKET_BIT_SIZE) {
            // std::cerr << "offset = " << offset << std::endl;
            std::swap(in_arr, out_arr);
            // debug_mem(*in_arr, "in_arr");
            ocl_radixSort01LocalCounting.exec(workSize, *in_arr, pref_sum_buf_2, n, offset);
            // debug_mem(pref_sum_buf_2, "pref_sum_buf_2 (local_buckets)");

            unsigned int cur_n = BUCKET_SIZE * GROUPS_COUNT;
            unsigned int pow2 = 1;

            ocl_fillBufferWithZeros.exec(
                gpu::WorkSize(GROUP_SIZE, BUCKET_SIZE * GROUPS_COUNT),
                global_pref_sums,
                BUCKET_SIZE * GROUPS_COUNT
            );

            while (pow2 <= BUCKET_SIZE * GROUPS_COUNT) {
                ocl_radixSort03GlobalPrefixesScanAccumulation.exec(
                    gpu::WorkSize(GROUP_SIZE, BUCKET_SIZE * GROUPS_COUNT),
                    pref_sum_buf_2,
                    global_pref_sums,
                    BUCKET_SIZE * GROUPS_COUNT,
                    pow2
                );

                std::swap(pref_sum_buf_2, pref_sum_buf_1);
                pow2 *= 2;
                if (pow2 <= BUCKET_SIZE * GROUPS_COUNT) {
                    ocl_radixSort02GlobalPrefixesScanSumReduction.exec(
                        gpu::WorkSize(GROUP_SIZE, (cur_n + 1) / 2),
                        pref_sum_buf_1,
                        pref_sum_buf_2,
                        cur_n
                    );
                    cur_n = (cur_n + 1) / 2;
                }
            }

            // debug_mem(global_pref_sums, "global_pref_sums");

            ocl_radixSort04Scatter.exec(
                gpu::WorkSize(GROUP_SIZE, n),
                *in_arr,
                global_pref_sums,
                *out_arr,
                n,
                offset
            );
            // debug_mem(*out_arr, "out_arr");
            // std::cerr << "------------ ---------------\n";
        }
        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << n / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_sorted = (*out_arr).readVector();

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
