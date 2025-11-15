#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>

std::tuple<std::vector<unsigned int>, std::vector<unsigned int>, std::vector<unsigned int>> generate_csr_matrix(
    unsigned int nrows, unsigned int ncols,
    unsigned int min_non_zero_values_per_row, unsigned int max_non_zero_values_per_row,
    unsigned int max_value)
{
    std::vector<unsigned int> csr_row_offsets(nrows + 1, 0);
    for (unsigned int row = 0; row < nrows; ++row) {
        FastRandom r(239 * row);
        unsigned int non_zero_row_values = r.next(min_non_zero_values_per_row, max_non_zero_values_per_row);
        csr_row_offsets[row + 1] = csr_row_offsets[row] + non_zero_row_values;
    }
    const unsigned int number_of_non_zero_values = csr_row_offsets[nrows];

    std::vector<unsigned int> csr_columns(number_of_non_zero_values, 0);
    std::vector<unsigned int> csr_values(number_of_non_zero_values, 0);

    #pragma omp parallel for
    for (int row = 0; row < nrows; ++row) {
        FastRandom r(239 * row);
        unsigned int non_zero_row_values = csr_row_offsets[row + 1] - csr_row_offsets[row];
        std::vector<unsigned int> columns = r.random_sorted_unique_values(0, ncols - 1, non_zero_row_values);

        for (int i = 0; i < non_zero_row_values; ++i) {
            unsigned int value = r.next(0, max_value);
            csr_columns[csr_row_offsets[row] + i] = columns[i];
            csr_values[csr_row_offsets[row] + i] = value;
        }
    }

    return {csr_row_offsets, csr_columns, csr_values};
}

std::vector<unsigned int> generate_vector(
    unsigned int n, unsigned int max_value=10000)
{
    FastRandom r(2391);

    std::vector<unsigned int> data(n, 0);
    for (unsigned int i = 0; i < n; ++i) {
        data[i] = r.next(0, max_value);
    }

    return data;
}

std::vector<unsigned int> sparse_csr_matrix_vector_multiplication(
    const std::vector<unsigned int> &csr_row_offsets, const std::vector<unsigned int> &csr_columns, const std::vector<unsigned int> &csr_values,
    const std::vector<unsigned int> &vector_values,
    unsigned int nrows, unsigned int ncols)
{
    std::vector<unsigned int> result_vector_values(nrows, 0);

    #pragma omp parallel for schedule(dynamic, 128)
    for (int row = 0; row < nrows; ++row) {
        size_t accumulator = 0;
        unsigned int row_from = csr_row_offsets[row];
        unsigned int row_to = csr_row_offsets[row + 1];
        for (unsigned int i = row_from; i < row_to; ++i) {
            unsigned int col = csr_columns[i];
            accumulator += csr_values[i] * vector_values[col];
        }
        result_vector_values[row] = narrow_cast<unsigned int>(accumulator);
    }

    return result_vector_values;
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

    ocl::KernelSource ocl_spvm(ocl::getSparseCSRMatrixVectorMult());

    avk2::KernelSource vk_spvm(avk2::getSparseCSRMatrixVectorMult());

    FastRandom r;

    const unsigned int nrows = 1000*1000; // TODO при отладке используйте минимальное n (например n=5 или n=10) при котором воспроизводится бага
    const unsigned int ncols = 1000*1000;
    const unsigned int max_value = 1000;
    std::cout << "Evaluating CSR matrix nrows x ncols=" << nrows << "x" << ncols << " with values in range [0; " << max_value << "]" << std::endl;

    std::vector<std::pair<unsigned int, unsigned int>> evaluated_min_max_nnz_per_row = {
        {32, 32},
        {128, 128},
        {1, 32},
        {1, 128},
        {32, 128},
    };
    for (auto [min_nnz_per_row, max_nnz_per_col]: evaluated_min_max_nnz_per_row) {
        std::cout << "____________________________________________________________________________________________" << std::endl;
        const auto [csr_row_offsets, csr_columns, csr_values] = generate_csr_matrix(nrows, ncols, min_nnz_per_row, max_nnz_per_col, max_value);
        const std::vector<unsigned int> vector_values = generate_vector(ncols, max_value);
        const unsigned int number_of_non_zero_values = csr_row_offsets[nrows];
        const unsigned int nnz = number_of_non_zero_values;

        std::vector<unsigned int> nnz_per_row(nrows);
        for (int row = 0; row < nrows; ++row) {
            nnz_per_row[row] = csr_row_offsets[row + 1] - csr_row_offsets[row];
        }

        std::cout << "Evaluating with NNZ per row in range [" << min_nnz_per_row << "; " << max_nnz_per_col << "], median NNZ per row=" << stats::median(nnz_per_row) << ", total NNZ=" << nnz << "..." << std::endl;

        timer t;
        const std::vector<unsigned int> cpu_results = sparse_csr_matrix_vector_multiplication(csr_row_offsets, csr_columns, csr_values, vector_values, nrows, ncols);
        // Вычисляем достигнутую эффективную пропускную способность видеопамяти
        // (из соображений что мы отработали идеально - считали один раз каждое ненулевое число из матрицы + из вектора + записали результаты)
        double memory_size_gb = sizeof(unsigned int) * (nnz + vector_values.size() + cpu_results.size()) / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU (multi-threaded via OpenMP) finished in " << t.elapsed() << " sec" << std::endl;
        std::cout << "CPU effective bandwidth: " << memory_size_gb / t.elapsed() << " GB/s (" << nnz / 1000 / 1000 / t.elapsed() << " uint millions/s)" << std::endl;

        // Аллоцируем буферы в VRAM
        gpu::gpu_mem_32u csr_row_offsets_gpu(nrows + 1), csr_columns_gpu(nnz), csr_values_gpu(nnz), vector_values_gpu(ncols), output_vector_values_gpu(nrows);

        // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
        csr_row_offsets_gpu.writeN(csr_row_offsets.data(), csr_row_offsets.size());
        csr_columns_gpu.writeN(csr_columns.data(), csr_columns.size());
        csr_values_gpu.writeN(csr_values.data(), csr_values.size());
        vector_values_gpu.writeN(vector_values.data(), vector_values.size());

        // Советую занулить (или еще лучше - заполнить какой-то уникальной константой, например 255) все буферы
        // В некоторых случаях это ускоряет отладку, но обратите внимание, что fill реализован через копию множества нулей по PCI-E, то есть он очень медленный
        // Если вам нужно занулять буферы в процессе вычислений - создайте кернел который это сделает
        output_vector_values_gpu.fill(255);

        // Запускаем кернел (несколько раз и с замером времени выполнения)
        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) { // TODO при отладке запускайте одну итерацию
            t.restart();

            // Запускаем кернел, с указанием размера рабочего пространства и передачей всех аргументов
            // Если хотите - можете удалить ветвление здесь и оставить только тот код который соответствует вашему выбору API
            if (context.type() == gpu::Context::TypeOpenCL) {
                ocl_spvm.exec(gpu::WorkSize(GROUP_SIZE, nrows), csr_row_offsets_gpu, csr_columns_gpu, csr_values_gpu, nrows, vector_values_gpu, output_vector_values_gpu);
            } else if (context.type() == gpu::Context::TypeCUDA) {
                // TODO
                throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            } else if (context.type() == gpu::Context::TypeVulkan) {
                // TODO
                throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
            } else {
                rassert(false, 4531412341, context.type());
            }

            times.push_back(t.elapsed());
        }

        std::cout << "GPU SpMV (sparse matrix-vector multiplication) times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили его переупорядоченным)
        std::cout << "GPU SpMV median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s (" << nnz / 1000 / 1000 / stats::median(times) << " uint millions/s)" << std::endl;

        // Считываем результат по PCI-E шине: GPU VRAM -> CPU RAM
        std::vector<unsigned int> gpu_results = output_vector_values_gpu.readVector();

        rassert(cpu_results.size() == gpu_results.size(), 43572348952, cpu_results.size(), gpu_results.size());

        // Сверяем результат
        for (size_t i = 0; i < cpu_results.size(); ++i) {
            rassert(cpu_results[i] == gpu_results[i], 566324523452323, cpu_results[i], gpu_results[i], i);
        }
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
