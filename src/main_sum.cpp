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

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    std::vector<double> times_pci;
    for (int i = 0; i < 5; i++) {
        timer timer;
        input_gpu.writeN(values.data(), n);
        times_pci.push_back(timer.elapsed());
    }
    double size_gb = sizeof(unsigned int) * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "PCI-E bandwidth: " << size_gb / stats::median(times_pci) << " GB/s" << std::endl;

    gpu::gpu_mem_32u a_gpu(n), b_gpu(n);
    a_gpu.writeN(values.data(), n);

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
        for (int iter = 0; iter < 1; ++iter) {
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
                        sum_accum_gpu.fill(0);
                        ocl_sum03LocalMemoryAtomicPerWorkgroup.exec(gpu::WorkSize(GROUP_SIZE, n), input_gpu, sum_accum_gpu, n);
                        sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "04 local reduction") {
                        int cnt = 0;
                        for (uint sz = n; sz > 1; cnt++) {
                            size_t num_groups = (sz + GROUP_SIZE - 1) / GROUP_SIZE;
                            size_t global = num_groups * GROUP_SIZE;
                            size_t local = GROUP_SIZE;

                            gpu::WorkSize ws(local, global);

                            if (cnt % 2 == 0) {
                                ocl_sum04LocalReduction.exec(ws, a_gpu, b_gpu, (uint)sz);
                            } else {
                                ocl_sum04LocalReduction.exec(ws, b_gpu, a_gpu, (uint)sz);
                            }

                            sz = num_groups;
                        }
                        if (cnt % 2 == 0)
                            a_gpu.readN(&gpu_sum, 1);
                        else
                            b_gpu.readN(&gpu_sum, 1);
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
