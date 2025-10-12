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
    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    FastRandom r;

    int n = 100*1000*1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
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

    unsigned int chunks_count = div_ceil(n, ELEMENTS_PER_CHUNK);

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n), output_gpu(n), output_buffer_gpu(n);
    //rassert((1 << BITS_PER_PASS) <= std::numeric_limits<unsigned char>::max(), 245234123321); // let's ensure that we will not overlfow counters
    gpu::gpu_mem_32u per_chunk_counts_gpu(chunks_count * CASES_PER_PASS);
    gpu::gpu_mem_32u buffer1_per_chunk_counts_gpu(chunks_count * CASES_PER_PASS);
    gpu::gpu_mem_32u buffer2_per_chunk_counts_gpu(chunks_count * CASES_PER_PASS);
    gpu::gpu_mem_32u per_chunk_counts_prefixes_gpu(chunks_count * CASES_PER_PASS);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - напишите кернел который это сделает
    output_gpu.fill(255);
    output_buffer_gpu.fill(255);
    per_chunk_counts_gpu.fill(255);
    buffer1_per_chunk_counts_gpu.fill(255);
    buffer2_per_chunk_counts_gpu.fill(255);
    per_chunk_counts_prefixes_gpu.fill(255);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeCUDA) {
            gpu::gpu_mem_32u *a_gpu = &input_gpu;
            gpu::gpu_mem_32u *b_gpu = &output_buffer_gpu;//&output_gpu;
            const int last_bits_offset = 28;
            for (int bits_offset = 0; bits_offset <= last_bits_offset; bits_offset += BITS_PER_PASS) {
//                std::cout << "__________________bits offset: " << bits_offset << "_______________________________________________________" << std::endl;
                // 1) in each chunk calculate combs counts
                cuda::radix_sort_01_local_counting(gpu::WorkSize(WG_SIZE, div_ceil(n, ELEMENTS_PER_WORK_ITEM)), *a_gpu, per_chunk_counts_gpu, bits_offset, n);
//                std::cout << "                        a_gpu: " << stats::vectorToString(a_gpu->readVector(), 64) << std::endl;
//                std::cout << "                        a_gpu: " << stats::vectorToString(debug::prettyBits(a_gpu->readVector(), max_value, bits_offset, BITS_PER_PASS), 64) << std::endl;
//                std::cout << "         per_chunk_counts_gpu: " << stats::vectorToString(debug::toInt(per_chunk_counts_gpu.readVector()), 64) << std::endl;
                // 2) prefix sum to estimate global offsets for each chunk combs

                {
                    unsigned int maximal_pow2 = 0;
                    while ((1 << maximal_pow2) <= chunks_count) {
                        ++maximal_pow2;
                    }
                    //cuda::fill_buffer_with_zeros(gpu::WorkSize(GROUP_SIZE, chunks_count * CASES_PER_PASS), per_chunk_counts_prefixes_gpu, chunks_count * CASES_PER_PASS);
                    for (unsigned int pow2 = 0; pow2 <= maximal_pow2; ++pow2) {
                        auto prev_pow2_sum_gpu = (pow2 % 2 == 0) ? buffer1_per_chunk_counts_gpu : buffer2_per_chunk_counts_gpu;
                        auto cur_pow2_sum_gpu = (pow2 % 2 == 0) ? buffer2_per_chunk_counts_gpu : buffer1_per_chunk_counts_gpu;
                        if (pow2 == 0) {
                            cur_pow2_sum_gpu = per_chunk_counts_gpu;
                        } else {
                            if (pow2 == 1) {
                                prev_pow2_sum_gpu = per_chunk_counts_gpu;
                            }
                            unsigned int prev_pow2 = pow2 - 1;
                            unsigned int pow2_sum_count = div_ceil(chunks_count, (1u << prev_pow2));
                            cuda::radix_sort_02_global_prefixes_scan_sum_reduction(gpu::WorkSize(GROUP_SIZE, pow2_sum_count * CASES_PER_PASS), prev_pow2_sum_gpu, cur_pow2_sum_gpu, pow2_sum_count);
                        }

                        cuda::radix_sort_03_global_prefixes_scan_accumulation(gpu::WorkSize(GROUP_SIZE, (chunks_count * CASES_PER_PASS)), cur_pow2_sum_gpu, per_chunk_counts_prefixes_gpu, chunks_count, pow2);

                        // TODO use n=10 + this to debug and look at data to understand what's going on:
//                                    std::cout << "_____________________pow2=" << pow2 << std::endl;
//                                    std::cout << "values  =" << stats::vectorToString(per_chunk_counts_gpu.readVector(), 32) << std::endl;
//                                    std::cout << "pow2_sum=" << stats::vectorToString(cur_pow2_sum_gpu.readVector(), 32) << std::endl;
//                                    std::cout << "prefix  =" << stats::vectorToString(per_chunk_counts_prefixes_gpu.readVector(), 32) << std::endl;
//                                    if (pow2 == 1) {
//                                        return;
//                                    }
                    }

                    std::vector<unsigned int> fast_prefixes = per_chunk_counts_prefixes_gpu.readVector();
                    //std::cerr << " fast prefixes: " << stats::vectorToString(fast_prefixes, 1024) << std::endl;
                }

//                std::cout << "per_chunk_counts_prefixes_gpu: " << stats::vectorToString(per_chunk_counts_prefixes_gpu.readVector(), 64) << std::endl;
                // 3) move chunks w.r.t. global+local offsets for each chunk combs
                cuda::radix_sort_04_scatter(gpu::WorkSize(WG_SIZE, div_ceil(n, ELEMENTS_PER_WORK_ITEM)), *a_gpu, per_chunk_counts_prefixes_gpu, *b_gpu, bits_offset, n);
//                std::cout << "                        b_gpu: " << stats::vectorToString(debug::prettyBits(b_gpu->readVector(), max_value, bits_offset, BITS_PER_PASS), 64) << std::endl;
//                std::cout << "                        b_gpu: " << stats::vectorToString(b_gpu->readVector(), 64) << std::endl;
                if (bits_offset == last_bits_offset) {
                    rassert(b_gpu == &output_gpu, 4365214321); // ensure that reordered values are in the buffer we expected it to be
                } else {
                    if (a_gpu == &input_gpu) {
                        a_gpu = &output_gpu;//&output_buffer_gpu;
                    }
                    std::swap(a_gpu, b_gpu);
                }
            }
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
    std::vector<unsigned int> gpu_sorted = output_gpu.readVector();

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
