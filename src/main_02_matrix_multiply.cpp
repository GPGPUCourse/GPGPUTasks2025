#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include <libgpu/vulkan/vk/common_host.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <iomanip>

namespace cpu {
void multiply(
    const std::vector<float> &a,
    const std::vector<float> &b,
          std::vector<float> &c,
                 unsigned int w,
                 unsigned int h,
                 unsigned int k,
                  bool with_omp)
{
    #pragma omp parallel for schedule(dynamic, 1) if (with_omp)
    for (ptrdiff_t j = 0; j < h; ++j) {
        for (ptrdiff_t i = 0; i < w; ++i) {
            float acc = 0.0f;

            for (int ki = 0; ki < k; ++ki) {
                acc += a[j * k + ki] * b[ki * w + i];
            }

            c[j * w + i] = acc;
        }
    }
}
}

void run(int argc, char** argv)
{
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

    ocl::KernelSource ocl_matrix03MultiplyNaive(ocl::getMatrix03MultiplyNaive());
    ocl::KernelSource ocl_matrix04MultiplyViaLocalMemory(ocl::getMatrix04MultiplyViaLocalMemory());

    avk2::KernelSource vk_matrix03MultiplyNaive(avk2::getMatrix03MultiplyNaive());
    avk2::KernelSource vk_matrix04MultiplyViaLocalMemory(avk2::getMatrix04MultiplyViaLocalMemory());
    avk2::KernelSource vk_matrix05MultiplyCooperativeMatrix(avk2::getMatrix05MultiplyCooperativeMatrix());

    unsigned int ksize = 128;
    unsigned int w = ksize * 32;
    unsigned int k = ksize * 8;
    unsigned int h = ksize * 16;
    std::cout << "C = A x B, matrices size: C (rows=H=" << h << " x cols=W=" << w << ")"
              << " = A (rows=H=" << h << " x cols=K=" << k << ") x B (rows=K=" << k << " x cols=W=" << w << ")" << std::endl;
    std::cout << "matrices data size: A - " << sizeof(float) * h * k / 1024 / 1024 << " MB, B - " << sizeof(float) * k * w / 1024 / 1024 << " MB, C - " << sizeof(float) * k * w / 1024 / 1024 << " MB" << std::endl;

    std::vector<float> input_a_cpu(h * k, 0);  // rows=H x cols=K
    std::vector<float> input_b_cpu(k * w, 0);  // rows=K x cols=W
    std::vector<float> output_c_cpu(h * w, 0); // rows=H x cols=W
    std::vector<float> output_c_gpu(h * w, 0); // rows=H x cols=W
    FastRandom r;
    for (size_t i = 0; i < input_a_cpu.size(); ++i) {
        input_a_cpu[i] = r.nextf();
    }
    for (size_t i = 0; i < input_b_cpu.size(); ++i) {
        input_b_cpu[i] = r.nextf();
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32f matrix_a_gpu(h * k); // rows=H x cols=K
    gpu::gpu_mem_32f matrix_b_gpu(k * w); // rows=K x cols=W
    gpu::gpu_mem_32f matrix_c_gpu(h * w); // rows=H x cols=W

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    matrix_a_gpu.writeN(input_a_cpu.data(), input_a_cpu.size());
    matrix_b_gpu.writeN(input_b_cpu.data(), input_b_cpu.size());

    std::vector<std::string> algorithm_names = {
        "CPU with OpenMP",
        "01 naive",
        "02 using local memory",
    };

    // TODO 020 Это добровольное задание за супер-пупер-баллы престижа сверх нормы
    bool I_Want_Super_Puper_Prestige_Points = false;
    if (I_Want_Super_Puper_Prestige_Points) {
        if (context.type() == gpu::Context::TypeCUDA) {
            algorithm_names.push_back("03 using WMMA (Tensor Cores) [+Prestige Points]");
        }
        if (context.type() == gpu::Context::TypeVulkan) {
            rassert(context.vk()->device().supportsExtension("VK_KHR_cooperative_matrix"), 32452365324632);
            auto device_supported_cooperative_matrix_sizes = context.vk()->device().supportedCooperativeMatrixSizes();
            rassert(context.vk()->device().isCooperativeMatrixSizeSupported(DataType16f, DataType32f, 16, 16, 16), 235243524356);
            algorithm_names.push_back("03 using cooperative matrix [+Prestige Points]");
        }
    }

    for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {
        const std::string& algorithm = algorithm_names[algorithm_index];
        std::cout << "______________________________________________________" << std::endl;
        std::cout << "Evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

        // Запускаем алгоритм (несколько раз и с замером времени выполнения)
        std::vector<double> times;
        int iters_count = (algorithm == "CPU with OpenMP") ? 1 : 10; // CPU is too slow
        for (int iter = 0; iter < iters_count; ++iter) {
            timer t;

            if (algorithm == "CPU with OpenMP") {
                cpu::multiply(input_a_cpu, input_b_cpu, output_c_cpu, w, h, k, true);
            } else {
                // _______________________________OpenCL_____________________________________________
                if (context.type() == gpu::Context::TypeOpenCL) {
                    if (algorithm == "01 naive") {
                        ocl_matrix03MultiplyNaive.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else if (algorithm == "02 using local memory") {
                        ocl_matrix04MultiplyViaLocalMemory.exec(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else {
                        rassert(false, 7652345234321, algorithm, algorithm_index);
                    }
                    // _______________________________CUDA___________________________________________
                } else if (context.type() == gpu::Context::TypeCUDA) {
                    if (algorithm == "01 naive") {
                        cuda::matrix_multiply_naive(gpu::WorkSize(GROUP_SIZE, 1, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else if (algorithm == "02 using local memory") {
                        cuda::matrix_multiply_via_local_memory(gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else if (algorithm == "03 using WMMA (Tensor Cores) [+Prestige Points]") {
                        cuda::matrix_multiply_wmma(gpu::WorkSize(16, 2, w, h * 2 / 16), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else {
                        rassert(false, 652345234321, algorithm, algorithm_index);
                    }
                    // _______________________________Vulkan_________________________________________
                } else if (context.type() == gpu::Context::TypeVulkan) {
                    struct {
                        unsigned int w;
                        unsigned int h;
                        unsigned int k;
                    } params = {w, h, k};
                    if (algorithm == "01 naive") {
//                        vk_matrix03MultiplyNaive.exec(params, gpu::WorkSize(1, 1, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu);
                    } else if (algorithm == "02 using local memory") {
//                        vk_matrix04MultiplyViaLocalMemory.exec(params, gpu::WorkSize(1, 1, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu);
                    } else if (algorithm == "03 using cooperative matrix [+Prestige Points]") {
                        vk_matrix05MultiplyCooperativeMatrix.exec(params, gpu::WorkSize(VK_SUBGROUP_SIZE, 1, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu);
                    } else {
                        rassert(false, 7652345234321, algorithm, algorithm_index);
                    }
                } else {
                    rassert(false, 546345243, context.type());
                }
            }

            times.push_back(t.elapsed());
        }
        std::cout << "algorithm times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        // Вычисляем достигнутую эффективную пропускную способность алгоритма
        double total_ops = 1.0 * h * w * (k + k - 1); // общее число сложений и умножений
        double gflops = 1000*1000*1000;
        std::cout << "algorithm GFlops: " << total_ops / gflops / stats::median(times) << " GFlops" << std::endl;
        std::cout << "algorithm effective memory bandwidth: " << 1.0 * (h * k + k * w + h * w) * sizeof(float) / 1024 / 1024 / 1024 / stats::median(times) << " GB/s" << std::endl;

        // Сверяем результат
        if (algorithm != "CPU with OpenMP") {
            std::vector<float> results = matrix_c_gpu.readVector();
            std::vector<float> relative_errors;
            for (size_t j = 0; j < h; ++j) {
                for (size_t i = 0; i < w; ++i) {
                    float gpu_value = results[j * w + i];
                    float cpu_value = output_c_cpu[j * w + i];
                    float error = std::abs(gpu_value - cpu_value);
                    float relative_error = error / std::abs(cpu_value);
                    relative_errors.push_back(relative_error);
                }
            }
            std::cout << "relative differences with CPU: " << stats::valuesStatsLine(relative_errors) << std::endl;
            float median_relative_error = stats::median(relative_errors);
            float perc99_relative_error = stats::percentile(relative_errors, 99);
            std::cout << "median relative difference with CPU: " << median_relative_error << std::endl;
            std::cout << "99% percentile relative difference with CPU: " << perc99_relative_error << std::endl;
            rassert(median_relative_error < 1e-3f, 15321452412431, median_relative_error);
            rassert(perc99_relative_error < 1e-1f, 54623452334232, perc99_relative_error);
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