#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>

void PrintVector(const std::vector<unsigned int>& arr)
{
    std::cout << "{";
    for (auto x : arr)
        std::cout << x << ", ";
    std::cout << "}";
}

void PrintGPUArray(const gpu::gpu_mem_32u& arr)
{
    std::vector<unsigned int> arr_cpu = arr.readVector();
    PrintVector(arr_cpu);
}

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_fill_with_zeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_sum_reduction(ocl::getPrefixSum01Reduction());
    ocl::KernelSource ocl_prefix_accumulation(ocl::getPrefixSum02PrefixAccumulation());

    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    size_t total_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        as[i] = (3 * (i + 5) + 7) % 17;
        total_sum += as[i];
        rassert(total_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, total_sum, as[i], i); // ensure no overflow
    }

    // std::cout << "\noriginal vector: ";
    // PrintVector(as);
    // std::cout << "\n\n";

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n), buffer1_pow2_sum_gpu(n), buffer2_pow2_sum_gpu(n), prefix_sum_accum_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        ocl_fill_with_zeros.exec(gpu::WorkSize(GROUP_SIZE, n), prefix_sum_accum_gpu, n);
        ocl_prefix_accumulation.exec(gpu::WorkSize(GROUP_SIZE, n), input_gpu, prefix_sum_accum_gpu, n, 0);

        // std::cout << "\nprefix_sum_accum_gpu: ";
        // PrintGPUArray(prefix_sum_accum_gpu);
        // std::cout << "\n\n";

        for (uint k = 1, window_size = 2; window_size <= n; k++, window_size <<= 1) {
            if (k == 1) {
                ocl_sum_reduction.exec(gpu::WorkSize(GROUP_SIZE, (n + 1) / 2), input_gpu, buffer1_pow2_sum_gpu, n);
                ocl_prefix_accumulation.exec(gpu::WorkSize(GROUP_SIZE, n), buffer1_pow2_sum_gpu, prefix_sum_accum_gpu, n, k);
            } else if (k % 2 == 0) {
                ocl_sum_reduction.exec(gpu::WorkSize(GROUP_SIZE, (n + 1) / 2), buffer1_pow2_sum_gpu, buffer2_pow2_sum_gpu, div_ceil(n, window_size >> 1));
                ocl_prefix_accumulation.exec(gpu::WorkSize(GROUP_SIZE, n), buffer2_pow2_sum_gpu, prefix_sum_accum_gpu, n, k);
            } else {
                ocl_sum_reduction.exec(gpu::WorkSize(GROUP_SIZE, (n + 1) / 2), buffer2_pow2_sum_gpu, buffer1_pow2_sum_gpu, div_ceil(n, window_size >> 1));
                ocl_prefix_accumulation.exec(gpu::WorkSize(GROUP_SIZE, n), buffer1_pow2_sum_gpu, prefix_sum_accum_gpu, n, k);
            }

            // if (k % 2 == 1) {
            //     std::cout << "window size: " << window_size << "\tbuffer: ";
            //     PrintGPUArray(buffer1_pow2_sum_gpu);
            //     std::cout << "\n";
            // } else {
            //     std::cout << "window size: " << window_size << "\tbuffer: ";
            //     PrintGPUArray(buffer2_pow2_sum_gpu);
            //     std::cout << "\n";
            // }

            // std::cout << "window size: " << window_size << "\tprefix_sum_accum_gpu: ";
            // PrintGPUArray(prefix_sum_accum_gpu);
            // std::cout << "\n";
        }

        times.push_back(t.elapsed());
    }
    std::cout << "prefix sum times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили префиксные суммы)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "prefix sum median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_prefix_sum = prefix_sum_accum_gpu.readVector();

    // Сверяем результат
    size_t cpu_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        cpu_sum += as[i];
        rassert(cpu_sum == gpu_prefix_sum[i], 566324523452323, cpu_sum, gpu_prefix_sum[i], i);
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
