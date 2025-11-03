#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/fast_random.h>
#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"
#include "libgpu/work_size.h"

void merge_sort(int n, gpu::gpu_mem_32u input, gpu::gpu_mem_32u output, ocl::KernelSource& local_sort, ocl::KernelSource& merge)
{
    gpu::gpu_mem_32u buffer(n);
    input.copyToN(buffer, n);
    // 1. Sorting within one group to make some baseline
    local_sort.exec(gpu::WorkSize(GROUP_SIZE, n), buffer, n);
    // 2. Iterative merging
    bool forward = true;
    for (int block_size = GROUP_SIZE; block_size < n; block_size <<= 1, forward = !forward) {
        gpu::gpu_mem_32u in = forward ? buffer : output;
        gpu::gpu_mem_32u out = forward ? output : buffer;
        int merge_block = block_size << 1;
        int groups_per_merge = (merge_block + GROUP_SIZE - 1) / GROUP_SIZE;
        int num_merges = (n + merge_block - 1) / merge_block;
        int total_groups = groups_per_merge * num_merges;
        int global_size = total_groups * GROUP_SIZE;
        merge.exec(gpu::WorkSize(GROUP_SIZE, global_size), in, out, n, block_size);
    }
    if (forward) {
        buffer.copyToN(output, n);
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

    ocl::KernelSource ocl_mergeLocalSort(ocl::getMergeLocalSort());
    ocl::KernelSource ocl_mergeMerge(ocl::getMergeMerge());

    FastRandom r;

    int n = 100 * 1000 * 1000;
    int min_value = 1; // это сделано для упрощения, чтобы существовало очевидное -INFINITY значение
    int max_value = std::numeric_limits<int>::max() - 1;
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(min_value, max_value);
    }
    std::cout << "n=" << n << " values in range [" << min_value << "; " << max_value << "]" << std::endl;

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
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / t.elapsed() << " GB/s (" << static_cast<double>(n) / 1000 / 1000 / t.elapsed() << " uint millions/s)" << std::endl;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u output_gpu(n);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        merge_sort(n, input_gpu, output_gpu, ocl_mergeLocalSort, ocl_mergeMerge);

        times.push_back(t.elapsed());
    }
    std::cout << "GPU merge-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU merge-sort median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << static_cast<double>(n) / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

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
