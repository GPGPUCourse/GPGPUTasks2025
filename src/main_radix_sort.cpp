#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h" // TODO очень советую использовать debug::prettyBits(...) для отладки
#include "libgpu/shared_device_buffer.h"
#include "libgpu/work_size.h"

#include <fstream>


std::string to_binary_string(uint32_t value, unsigned int bits) {
    std::string result;
    for (unsigned int i = 0; i < bits; ++i) {
        result = ((value & 1) ? '1' : '0') + result;
        value >>= 1;
    }
    return result;
}

void print_vec(const std::vector<unsigned int>& v, bool make_binary = false, unsigned int bits = 3) {
    std::cout << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::setw(4);
        std::cout << (make_binary ? to_binary_string(v[i], bits) : std::to_string(v[i]));
        std::cout << " ";
    }
    std::cout << "]";
    std::cout << std::endl;
}

void run(int argc, char** argv) {
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

    int n = 100*1000*1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value << std::endl;

    // {
    //     // убедимся что в массиве есть хотя бы несколько повторяющихся значений
    //     size_t force_duplicates_attempts = 3;
    //     bool all_attempts_missed = true;
    //     for (size_t k = 0; k < force_duplicates_attempts; ++k) {
    //         size_t i = r.next(0, n - 1);
    //         size_t j = r.next(0, n - 1);
    //         if (i != j) {
    //             as[j] = as[i];
    //             all_attempts_missed = false;
    //         }
    //     }
    //     rassert(!all_attempts_missed, 4353245123412);
    // }

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
    // TODO это просто шаблонка, можете переименовать эти буферы, сделать другого размера/типа, удалить часть, добавить новые
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u input_copy(n);
    gpu::gpu_mem_32u buffer1_gpu(n),
                     buffer2_gpu(n),
                     buffer3_gpu(n),
                     buffer4_gpu(n),
                     buffer5_gpu(n);
    gpu::gpu_mem_32u buffer_output_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);
    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    input_gpu.copyToN(input_copy, n);
    buffer1_gpu.fill(255);
    buffer2_gpu.fill(255);
    buffer3_gpu.fill(255);
    buffer4_gpu.fill(255);
    buffer5_gpu.fill(255);
    buffer_output_gpu.fill(255);

    gpu::WorkSize work_size(GROUP_SIZE, n);


    auto prefix_sum = [
        &ocl_fillBufferWithZeros, work_size,
        &ocl_radixSort02GlobalPrefixesScanSumReduction,
        &ocl_radixSort03GlobalPrefixesScanAccumulation
    ](
        gpu::gpu_mem_32u& input,
        gpu::gpu_mem_32u& buffer_from,
        gpu::gpu_mem_32u& buffer_to,
        gpu::gpu_mem_32u& output,
        unsigned int n
    ) {
        ocl_fillBufferWithZeros.exec(work_size, output, n);
        
        unsigned int k = 1;
        unsigned int p = 0;
        gpu::gpu_mem_32u* from = &input;
        gpu::gpu_mem_32u* to = &buffer_from;

        while (k <= n) {
            ocl_radixSort02GlobalPrefixesScanSumReduction.exec(work_size, *from, *to, n, p);
            ocl_radixSort03GlobalPrefixesScanAccumulation.exec(work_size, *to, output, n, p);

            if (p == 0) {
                from = &buffer_from;
                to = &buffer_to;
            } else {
                std::swap(from, to);
            }
            k *= 2;
            p++;
        }
    };

    // std::cout << "input:\n";
    // print_vec(input_gpu.readVector());
    // print_vec(input_gpu.readVector(), true);


    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {
            gpu::gpu_mem_32u* in  = &input_copy;
            gpu::gpu_mem_32u* out = &buffer_output_gpu;

            for (unsigned int bit = 0; bit < 32 /* 32 */; ++bit) {
                // std::cout << "======== bit " << bit << " ========\n";
                //out->fill(255);

                // count zeros and ones
                gpu::gpu_mem_32u& zeros = buffer1_gpu;
                gpu::gpu_mem_32u& ones  = buffer2_gpu;
                ocl_radixSort01LocalCounting.exec(work_size, *in, zeros, ones, bit, n);

                // std::cout << "zeros cnt:\n";
                // print_vec(zeros.readVector());

                // std::cout << "ones cnt:\n";
                // print_vec(ones.readVector());

                // prefix sums for zeros
                gpu::gpu_mem_32u& zeros_sum = buffer5_gpu;
                prefix_sum(zeros, buffer3_gpu, buffer4_gpu, zeros_sum, n);

                // std::cout << "zeros prefix sum:\n";
                // print_vec(zeros_sum.readVector());

                unsigned int ones_offset = -1;
                zeros_sum.read(&ones_offset, sizeof(ones_offset), (n - 1) * sizeof(ones_offset));

                // std::cout << "offset for ones: " << ones_offset << std::endl;

                // scatter zeros
                ocl_radixSort04Scatter.exec(work_size, *in, *out, zeros_sum, bit, 0 /* is bit set */, 0 /* offset */, n);

                // std::cout << "scatter zeros:\n";
                // print_vec(out->readVector(), true);

                // prefix sums for ones
                gpu::gpu_mem_32u& ones_sum = buffer5_gpu;
                prefix_sum(ones, buffer3_gpu, buffer4_gpu, ones_sum, n);

                // std::cout << "ones prefix sum:\n";
                // print_vec(ones_sum.readVector());

                // scatter ones
                ocl_radixSort04Scatter.exec(work_size, *in, *out, ones_sum, bit, 1 /* is bit set */, ones_offset /* offset */, n);

                // std::cout << "scatter ones:\n";
                // print_vec(out->readVector(), true);
                
                std::swap(in, out);
            }
            
            if (out != &buffer_output_gpu) {
                // если нечетное число проходов - нужно скопировать результат в нужный буфер
                out->copyToN(buffer_output_gpu, n);
            }

            // std::cout << "result:\n";
            // print_vec(buffer_output_gpu.readVector());

            // std::cout << "input should not be changed:\n";
            // print_vec(input_gpu.readVector());
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
