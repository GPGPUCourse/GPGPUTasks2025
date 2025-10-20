#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

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

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    // ocl::KernelSource ocl_fill_with_zeros(ocl::getFillBufferWithZeros());
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

    // Некоторые размеры
    unsigned int l1_size = n, // ~100 миллионов
        l2_size = (l1_size + CHUNK_SIZE - 1) / CHUNK_SIZE, // ~195 тыс
        l3_size = (l2_size + CHUNK_SIZE - 1) / CHUNK_SIZE, // 381
        l4_size = (l3_size + CHUNK_SIZE - 1) / CHUNK_SIZE; // 1

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u
        input_gpu(n), // входные данные, ~100 миллионов
        final_sums_l1(n), // здесь будет результат
        partial_sums_l1(l1_size), // тут группы, в каждой рассчитана её собственная префиксная сумма
        group_sums_l1(l2_size), // тут для каждой группы лежит её сумма, ~195 тыс элементов

        final_sums_l2(l2_size),
        partial_sums_l2(l2_size), // и аналогично считаем префиксные суммы среди сумм групп
        group_sums_l2(l3_size), // делим l2 на чанки, для каждого чанка считаем префиксные суммы

        final_sums_l3(l3_size), // и ещё раз
        partial_sums_l3(l3_size),
        group_sums_l3(l4_size),

        partial_sums_l4(l4_size), // наконец-то закончили
        group_sums_l4(l4_size);

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(as.data(), n);

    auto get_thread_count = [](size_t n_elements) {
        return (n_elements + ELEM_PER_THREAD - 1) / ELEM_PER_THREAD;
    };

    // Запускаем кернел (несколько раз и с замером времени выполнения)
    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        // Подъем
        ocl_sum_reduction.exec(
            gpu::WorkSize(GROUP_SIZE, l2_size * GROUP_SIZE),
            input_gpu, partial_sums_l1, group_sums_l1, l1_size);
        ocl_sum_reduction.exec(
            gpu::WorkSize(GROUP_SIZE, l3_size * GROUP_SIZE),
            group_sums_l1, partial_sums_l2, group_sums_l2, l2_size);
        ocl_sum_reduction.exec(
            gpu::WorkSize(GROUP_SIZE, l4_size * GROUP_SIZE),
            group_sums_l2, partial_sums_l3, group_sums_l3, l3_size);
        ocl_sum_reduction.exec(
            gpu::WorkSize(GROUP_SIZE, l4_size * GROUP_SIZE),
            group_sums_l3, partial_sums_l4, group_sums_l4, l4_size);

        // Спуск
        ocl_prefix_accumulation.exec(
            gpu::WorkSize(GROUP_SIZE, get_thread_count(l3_size)),
            partial_sums_l3, partial_sums_l4, final_sums_l3, l3_size);
        ocl_prefix_accumulation.exec(
            gpu::WorkSize(GROUP_SIZE, get_thread_count(l2_size)),
            partial_sums_l2, final_sums_l3, final_sums_l2, l2_size);
        ocl_prefix_accumulation.exec(
            gpu::WorkSize(GROUP_SIZE, get_thread_count(l1_size)),
            partial_sums_l1, final_sums_l2, final_sums_l1, l1_size);

        times.push_back(t.elapsed());
    }
    std::cout << "prefix sum times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили префиксные суммы)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "prefix sum median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
    std::vector<unsigned int> gpu_prefix_sum = final_sums_l1.readVector();

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
