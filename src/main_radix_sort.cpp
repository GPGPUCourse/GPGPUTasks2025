#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include "debug.h"

#include <fstream>
#include <vector>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

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
    gpu::gpu_mem_32u buffer_a(n);
    gpu::gpu_mem_32u buffer_b(n);
    
    gpu::gpu_mem_32u prefix_inverted_gpu(n);
    gpu::gpu_mem_32u scan_buffer1(n);
    gpu::gpu_mem_32u scan_buffer2(n);

    input_gpu.writeN(as.data(), n);
    buffer_a.writeN(as.data(), n);
    buffer_b.fill(255);

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
        timer t;

        if (context.type() == gpu::Context::TypeOpenCL) {
            gpu::gpu_mem_32u* input_buf = &buffer_a;
            gpu::gpu_mem_32u* output_buf = &buffer_b;
            
            gpu::WorkSize workSize(GROUP_SIZE, n);
            
            for (unsigned int bit_position = 0; bit_position < 32; ++bit_position) {
                ocl_radixSort01LocalCounting.exec(workSize,
                    *input_buf,
                    prefix_inverted_gpu,
                    (unsigned int)n,
                    bit_position);
                
                gpu::gpu_mem_32u* reduction_input = &prefix_inverted_gpu;
                gpu::gpu_mem_32u* reduction_output = &scan_buffer1;
                gpu::gpu_mem_32u* reduction_temp = &scan_buffer2;
                
                unsigned int current_size = n;
                unsigned int level = 0;
                bool first_iteration = true;
                
                while (current_size > 1) {
                    unsigned int next_size = (current_size + 1) / 2;
                    
                    ocl_radixSort02GlobalPrefixesScanSumReduction.exec(
                        gpu::WorkSize(GROUP_SIZE, next_size),
                        *reduction_input,
                        *reduction_output,
                        current_size);
                    
                    ocl_radixSort03GlobalPrefixesScanAccumulation.exec(
                        workSize,
                        prefix_inverted_gpu,
                        (unsigned int)n,
                        level);
                    
                    if (first_iteration) {
                        reduction_input = reduction_output;
                        reduction_output = reduction_temp;
                        first_iteration = false;
                    } else {
                        std::swap(reduction_input, reduction_output);
                    }
                    current_size = next_size;
                    level++;
                }
                
                unsigned int count0;
                prefix_inverted_gpu.readN(&count0, 1, n - 1);
                
                ocl_radixSort04Scatter.exec(workSize,
                    *input_buf,
                    prefix_inverted_gpu,
                    *output_buf,
                    (unsigned int)n,
                    bit_position,
                    count0);
                
                std::swap(input_buf, output_buf);
            }
        } else if (context.type() == gpu::Context::TypeCUDA) {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
        } else if (context.type() == gpu::Context::TypeVulkan) {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
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
    std::vector<unsigned int> gpu_sorted = buffer_a.readVector();

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
            return 0;
        } if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            return 0;
        } else {
            return 1;
        }
    }

    return 0;
}
