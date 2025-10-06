#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <iomanip>

unsigned int cpu::sum(const unsigned int* values, unsigned int n)
{
    unsigned int sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += values[i];
    }
    return sum;
}

unsigned int cpu::sumOpenMP(const unsigned int* values, unsigned int n)
{
    unsigned int sum = 0;
    #pragma omp parallel for schedule(dynamic, 1024) reduction(+ : sum)
    for (ptrdiff_t i = 0; i < n; ++i) {
        sum += values[i];
    }
    return sum;
}

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_sum01Atomics(ocl::getSum01Atomics());
    ocl::KernelSource ocl_sum02AtomicsLoadK(ocl::getSum02AtomicsLoadK());
    ocl::KernelSource ocl_sum03LocalMemoryAtomicPerWorkgroup(ocl::getSum03LocalMemoryAtomicPerWorkgroup());
    ocl::KernelSource ocl_sum04LocalReduction(ocl::getSum04LocalReduction());

    avk2::KernelSource vk_sum01Atomics(avk2::getSum01Atomics());
    avk2::KernelSource vk_sum02AtomicsLoadK(avk2::getSum02AtomicsLoadK());
    avk2::KernelSource vk_sum03LocalMemoryAtomicPerWorkgroup(avk2::getSum03LocalMemoryAtomicPerWorkgroup());
    avk2::KernelSource vk_sum04LocalReduction(avk2::getSum04LocalReduction());

    unsigned int n = 100 * 1000 * 1000;
    rassert(n % LOAD_K_VALUES_PER_ITEM == 0, 4356345432524); // for simplicity
    std::vector<unsigned int> values(n, 0);
    size_t cpu_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        values[i] = (3 * (i + 5) + 7) % 17;
        cpu_sum += values[i];
        rassert(cpu_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, cpu_sum, values[i], i); // ensure no overflow
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u sum_accum_gpu(1);
    gpu::gpu_mem_32u reduction_buffer1_gpu(div_ceil(n, (unsigned int)GROUP_SIZE));
    gpu::gpu_mem_32u reduction_buffer2_gpu(div_ceil(n, (unsigned int)GROUP_SIZE));

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    input_gpu.writeN(values.data(), n);
    // TODO 1) замерьте здесь какая достигнута пропускная пособность PCI-E шины
    // TODO 2) сделайте замер хотя бы три раза
    // TODO 3) и выведите рассчет на основании медианного времени (в легко понятной форме - GB/s)

    std::vector<std::string> algorithm_names = {
        "CPU",
        "CPU with OpenMP",
        "01 atomicAdd from each workItem",
        "02 atomicAdd but each workItem loads K values",
        "03 local memory and atomicAdd from master thread",
        "04 local reduction",
    };

    for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {
        const std::string& algorithm = algorithm_names[algorithm_index];
        std::cout << "______________________________________________________" << std::endl;
        std::cout << "Evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

        // Запускаем алгоритм (несколько раз и с замером времени выполнения)
        std::vector<double> times;
        unsigned int gpu_sum = 0;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            if (algorithm == "CPU") {
                gpu_sum = cpu::sum(values.data(), n);
            } else if (algorithm == "CPU with OpenMP") {
                gpu_sum = cpu::sumOpenMP(values.data(), n);
            } else {
                // _______________________________OpenCL_____________________________________________
                if (context.type() == gpu::Context::TypeOpenCL) {
                    if (algorithm == "01 atomicAdd from each workItem") {
                        sum_accum_gpu.fill(0);
                        ocl_sum01Atomics.exec(gpu::WorkSize(GROUP_SIZE, n), input_gpu, sum_accum_gpu, n);
                        sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "02 atomicAdd but each workItem loads K values") {
                        sum_accum_gpu.fill(0);
                        ocl_sum02AtomicsLoadK.exec(gpu::WorkSize(GROUP_SIZE, n / LOAD_K_VALUES_PER_ITEM), input_gpu, sum_accum_gpu, n);
                        sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "03 local memory and atomicAdd from master thread") {
                        // TODO ocl_sum03LocalMemoryAtomicPerWorkgroup.exec(...);
                        throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                    } else if (algorithm == "04 local reduction") {
                        // TODO ocl_sum04LocalReduction.exec(...);
                        throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                    } else {
                        rassert(false, 652345234321, algorithm, algorithm_index);
                    }
                    // _______________________________CUDA___________________________________________
                } else if (context.type() == gpu::Context::TypeCUDA) {
                    if (algorithm == "01 atomicAdd from each workItem") {
                        sum_accum_gpu.fill(0);
                        cuda::sum_01_atomics(gpu::WorkSize(GROUP_SIZE, n), input_gpu, sum_accum_gpu, n);
                        sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "02 atomicAdd but each workItem loads K values") {
                        sum_accum_gpu.fill(0);
                        cuda::sum_02_atomics_load_k(gpu::WorkSize(GROUP_SIZE, n / LOAD_K_VALUES_PER_ITEM), input_gpu, sum_accum_gpu, n);
                        sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "03 local memory and atomicAdd from master thread") {
                        // TODO cuda::sum_03_local_memory_atomic_per_workgroup(...);
                        throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                    } else if (algorithm == "04 local reduction") {
                        // TODO cuda::sum_04_local_reduction(...);
                        throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                    } else {
                        rassert(false, 652345234321, algorithm, algorithm_index);
                    }
                    // _______________________________Vulkan_________________________________________
                } else if (context.type() == gpu::Context::TypeVulkan) {
                    if (algorithm == "01 atomicAdd from each workItem") {
                        sum_accum_gpu.fill(0);
                        vk_sum01Atomics.exec(n, gpu::WorkSize(GROUP_SIZE, n), input_gpu, sum_accum_gpu);
                        sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "02 atomicAdd but each workItem loads K values") {
                        sum_accum_gpu.fill(0);
                        vk_sum02AtomicsLoadK.exec(n, gpu::WorkSize(GROUP_SIZE, n / LOAD_K_VALUES_PER_ITEM), input_gpu, sum_accum_gpu);
                        sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "03 local memory and atomicAdd from master thread") {
                        // TODO vk_sum03LocalMemoryAtomicPerWorkgroup.exec(...);
                        throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                    } else if (algorithm == "04 local reduction") {
                        // TODO vk_sum04LocalReduction.exec(...);
                        throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
                    } else {
                        rassert(false, 652345234321, algorithm, algorithm_index);
                    }
                } else {
                    rassert(false, 546345243, context.type());
                }
            }

            times.push_back(t.elapsed());
        }
        std::cout << "algorithm times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        // Вычисляем достигнутую эффективную пропускную способность алгоритма (из соображений что мы отработали в один проход по входному массиву)
        double memory_size_gb = sizeof(unsigned int) * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "sum median effective algorithm bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

        // Сверяем результат
        rassert(cpu_sum == gpu_sum, 3452341235234456, cpu_sum, gpu_sum);

        // Проверяем что входные данные остались нетронуты (ведь мы их будем переиспользовать в других алгоритмах)
        std::vector<unsigned int> input_values = input_gpu.readVector();
        for (size_t i = 0; i < n; ++i) {
            rassert(input_values[i] == values[i], 6573452432, input_values[i], values[i]);
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
