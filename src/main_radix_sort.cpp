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

#define DEBUG 0

#if DEBUG
#define ITERS 1
// for debug only
void print_gpu_vector(const gpu::gpu_mem_32u& buf) {
    const std::vector<unsigned int> vec = buf.readVector();
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << "(" << i << ": " <<  vec[i] << ") ";
        if (i % 16 == 15) std::cout << '\n';
    }
    std::cout << std::endl;
}
#else
#define ITERS 10
#endif


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

    ocl::KernelSource ocl_copyBuf(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    FastRandom r;

#if DEBUG 
    std::cout << "DEBUG MODE ENABLED" << std::endl;
    const int n = 1024;
    const int max_value = 15;
#else
    const int n = 100*1000*1000;
    const int max_value = std::numeric_limits<int>::max(); // TODO при отладке используйте минимальное max_value (например max_value=8) при котором воспроизводится бага
#endif

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
        const double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << t.elapsed() << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / t.elapsed() << " GB/s (" << n / 1000 / 1000 / t.elapsed() << " uint millions/s)" << std::endl;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u buffer_output_gpu(n), buffer_output_gpu2(n);

    const unsigned int buffer_boxes_cnt = (n + GROUP_SIZE - 1) / GROUP_SIZE * NUM_BOXES;
    gpu::gpu_mem_32u buffer_boxes((n + GROUP_SIZE - 1) / GROUP_SIZE * NUM_BOXES);
    std::vector<gpu::gpu_mem_32u> reduc_buffers;
    std::vector<unsigned int> reduc_buffers_sizes;

    unsigned int sz = buffer_boxes_cnt;
    while (sz > 16u) {
        if ((sz & NUM_BOXES) == NUM_BOXES) sz += NUM_BOXES;
        sz >>= 1u;
        reduc_buffers_sizes.push_back(sz);
        reduc_buffers.push_back(gpu::gpu_mem_32u(sz));
    }

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
    // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
    // Если вам нужно занулять буферы в процессе вычислений - используйте кернел который это сделает (см. кернел fill_buffer_with_zeros)
    // buffer_output_gpu.fill(255);
    
    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < ITERS; ++iter) {
        timer t;
        if (context.type() != gpu::Context::TypeOpenCL) {
            rassert(false, 4531412341, context.type());
        }

        for (unsigned int shift = 0u; shift < 32u; shift += BITS_IN_RADIX_SORT_ITERATION) {
            ocl_radixSort01LocalCounting.exec(
                gpu::WorkSize(GROUP_SIZE, n),
                (shift == 0u ? input_gpu : buffer_output_gpu2),
                buffer_boxes,
                shift,
                n
            );
            
            ocl_radixSort02GlobalPrefixesScanSumReduction.exec(
                gpu::WorkSize(GROUP_SIZE, reduc_buffers_sizes[0]),
                buffer_boxes,
                reduc_buffers[0],
                buffer_boxes_cnt,
                reduc_buffers_sizes[0]
            );

            for (unsigned int i = 1; i < reduc_buffers.size(); ++i) {
                ocl_radixSort02GlobalPrefixesScanSumReduction.exec(
                    gpu::WorkSize(GROUP_SIZE, reduc_buffers_sizes[i]),
                    reduc_buffers[i - 1],
                    reduc_buffers[i],
                    reduc_buffers_sizes[i - 1],
                    reduc_buffers_sizes[i]
                );
            }

            for (int i = reduc_buffers.size() - 2; i >= 0; --i) {
                ocl_radixSort03GlobalPrefixesScanAccumulation.exec(
                    gpu::WorkSize(GROUP_SIZE, reduc_buffers_sizes[i]),
                    reduc_buffers[i],
                    reduc_buffers[i + 1],
                    reduc_buffers_sizes[i]
                );
            }

            ocl_radixSort03GlobalPrefixesScanAccumulation.exec(
                gpu::WorkSize(GROUP_SIZE, buffer_boxes_cnt),
                buffer_boxes,
                reduc_buffers[0],
                buffer_boxes_cnt
            );

            ocl_radixSort04Scatter.exec(
                gpu::WorkSize(GROUP_SIZE, n),
                (shift == 0 ? input_gpu : buffer_output_gpu2),
                buffer_boxes,
                buffer_output_gpu,
                reduc_buffers.back(),
                n,
                shift
            );

            if (shift < 28u) {
                ocl_copyBuf.exec(
                    gpu::WorkSize(GROUP_SIZE, n),
                    buffer_output_gpu2,
                    buffer_output_gpu,
                    n
                );
            }

        }

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    const double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
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
            return 0;
        } else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            return 0;
        } else {
            return 1;
        }
    }

    return 0;
}
