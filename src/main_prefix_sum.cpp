#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <memory>

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

    ocl::KernelSource ocl_reduce_block_sums(ocl::getReduceBlockSums());
    ocl::KernelSource ocl_scan_block_inclusive(ocl::getScanBlockInclusive());
    ocl::KernelSource ocl_add_block_offsets(ocl::getAddBlockOffsets());
    ocl::KernelSource ocl_fill_buffer_with_zeros(ocl::getFillBufferWithZeros());

    // avk2::KernelSource vk_fill_with_zeros(avk2::getFillBufferWithZeros());
    // avk2::KernelSource vk_sum_reduction(avk2::getPrefixSum01Reduction());
    // avk2::KernelSource vk_prefix_accumulation(avk2::getPrefixSum02PrefixAccumulation());

    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    size_t total_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        as[i] = (3 * (i + 5) + 7) % 17;
        total_sum += as[i];
        rassert(total_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, total_sum, as[i], i); // ensure no overflow
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n), prefix_sum_accum_gpu(n);

    // Precompute hierarchy once and preallocate all intermediate buffers to avoid reallocations
    const unsigned int tile = GROUP_SIZE * 2u;
    const unsigned int num_blocks = (n + tile - 1u) / tile; // ceil_div(n, tile)
    std::vector<unsigned int> level_counts;
    if (num_blocks > 0u) {
        level_counts.push_back(num_blocks);
        while (level_counts.back() > 1u) {
            const unsigned int curr_n = level_counts.back();
            const unsigned int next_n = (curr_n + tile - 1u) / tile;
            level_counts.push_back(next_n);
        }
    }

    // Per-level block-sum buffers (levels[i] stores sums of level i-1 tiles; level 0 is base per-tile sums)
    std::vector<std::unique_ptr<gpu::gpu_mem_32u>> level_sums;
    level_sums.reserve(level_counts.size());
    for (unsigned int cnt : level_counts) {
        level_sums.emplace_back(cnt ? std::make_unique<gpu::gpu_mem_32u>(cnt) : nullptr);
        if (cnt > 0u) {
            gpu::WorkSize ws_zero(GROUP_SIZE, cnt);
            ocl_fill_buffer_with_zeros.exec(ws_zero, *level_sums.back(), cnt);
        }
    }
    // Scratch buffers reused across iterations and levels
    // - scan_tmp: used as dummy per-element output in up-sweep, and as scan output in down-sweep
    // - scan_parent/scan_curr: ping-pong for down-sweep parent offsets
    // - dummy_next: placeholder for per-block sums during down-sweep scans
    std::unique_ptr<gpu::gpu_mem_32u> scan_tmp = num_blocks ? std::make_unique<gpu::gpu_mem_32u>(num_blocks) : nullptr;
    std::unique_ptr<gpu::gpu_mem_32u> scan_parent = num_blocks ? std::make_unique<gpu::gpu_mem_32u>(num_blocks) : nullptr;
    std::unique_ptr<gpu::gpu_mem_32u> scan_curr = num_blocks ? std::make_unique<gpu::gpu_mem_32u>(num_blocks) : nullptr;
    std::unique_ptr<gpu::gpu_mem_32u> dummy_next = num_blocks ? std::make_unique<gpu::gpu_mem_32u>(num_blocks) : nullptr;

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
        // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
        if (context.type() == gpu::Context::TypeOpenCL) {
            // Workgroup layout for base pass
            gpu::WorkSize ws_blocks(GROUP_SIZE, (size_t)num_blocks * GROUP_SIZE);

            // Base pass: per-block scan and gather base block sums into preallocated level_sums[0]
            if (num_blocks > 0u)
                ocl_scan_block_inclusive.exec(ws_blocks, input_gpu, prefix_sum_accum_gpu, *level_sums[0], n);

            // Up-sweep: build higher levels of block sums
            for (size_t li = 0; li + 1 < level_counts.size(); ++li) {
                const unsigned int curr_n = level_counts[li];
                const unsigned int next_n = level_counts[li + 1];
                gpu::WorkSize ws_next(GROUP_SIZE, (size_t)((next_n > 0u ? next_n : 1u) * GROUP_SIZE));

                if (next_n > 0u)
                    ocl_reduce_block_sums.exec(ws_next, *level_sums[li], *level_sums[li + 1], curr_n);
            }

            // Down-sweep: for each level, compute its scanned prefixes and add parent offsets
            // Use ping-pong between scan_parent and scan_curr to avoid reallocations
            gpu::gpu_mem_32u* parent_scan_ptr = nullptr;
            for (int li = (int)level_counts.size() - 1; li >= 0; --li) {
                const unsigned int curr_n = level_counts[(size_t)li];
                if (curr_n == 0u)
                    continue;
                const unsigned int curr_groups = curr_n;
                gpu::WorkSize ws_curr(GROUP_SIZE, (size_t)curr_groups * GROUP_SIZE);

                // Choose output buffer for this level scan
                gpu::gpu_mem_32u* curr_scan_ptr = parent_scan_ptr == scan_parent.get() ? scan_curr.get() : scan_parent.get();

                // Inclusive scan of level_sums[li] into curr_scan_ptr; write per-block sums to dummy_next (ignored)
                ocl_scan_block_inclusive.exec(ws_curr, *level_sums[(size_t)li], *curr_scan_ptr, *dummy_next, curr_n);

                // Accumulate parent offsets if not top level
                if (parent_scan_ptr)
                    ocl_add_block_offsets.exec(ws_curr, *parent_scan_ptr, *curr_scan_ptr, curr_n);

                // Apply block offsets to base per-element output at level 0
                if (li == 0 && num_blocks > 0u)
                    ocl_add_block_offsets.exec(ws_blocks, *curr_scan_ptr, prefix_sum_accum_gpu, n);

                parent_scan_ptr = curr_scan_ptr;
            }
        } else if (context.type() == gpu::Context::TypeCUDA) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // cuda::fill_buffer_with_zeros();
            // cuda::prefix_sum_01_sum_reduction();
            // cuda::prefix_sum_02_prefix_accumulation();
        } else if (context.type() == gpu::Context::TypeVulkan) {
            // TODO
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            // vk_fill_with_zeros.exec();
            // vk_sum_reduction.exec();
            // vk_prefix_accumulation.exec();
        } else {
            rassert(false, 4531412341, context.type());
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
