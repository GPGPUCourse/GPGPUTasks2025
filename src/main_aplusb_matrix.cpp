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

    // TODO 100 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 100 после этого реализуйте два кернела - максимально эффективный и максимально неэффктивный вариант сложения матриц - src/kernels/<ваш выбор>/aplusb_matrix_<bad/good>.<ваш выбор>
    // TODO 100 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DCUDA_SUPPORT=ON в CMake options
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_aplusb_matrix_bad(ocl::getAplusBMatrixBad());
    ocl::KernelSource ocl_aplusb_matrix_good(ocl::getAplusBMatrixGood());

    avk2::KernelSource vk_aplusb_matrix_bad(avk2::getAplusBMatrixBad());
    avk2::KernelSource vk_aplusb_matrix_good(avk2::getAplusBMatrixGood());

    unsigned int task_size = 64;
    unsigned int width = task_size * 256;
    unsigned int height = task_size * 128;
    std::cout << "matrices size: " << width << "x" << height << " = 3 * " << (sizeof(unsigned int) * width * height / 1024 / 1024) << " MB" << std::endl;


    std::vector<unsigned int> as(width * height, 0);
    std::vector<unsigned int> bs(width * height, 0);
    for (size_t i = 0; i < width * height; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    gpu::gpu_mem_32u a_gpu(width * height), b_gpu(width * height), c_gpu(width * height);
    a_gpu.write(as.data(), as.size() * sizeof(unsigned int));
    b_gpu.write(bs.data(), bs.size() * sizeof(unsigned int));

    {
        std::cout << "Running BAD matrix kernel..." << std::endl;

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            gpu::WorkSize workSize(16, 16, width, height);


            if (context.type() == gpu::Context::TypeOpenCL) {
                ocl_aplusb_matrix_bad.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
            } else if (context.type() == gpu::Context::TypeCUDA) {
                #if defined(CUDA_SUPPORT)
                    cuda::aplusb_matrix_bad(workSize, a_gpu, b_gpu, c_gpu, width, height);
                #endif            
            } else if (context.type() == gpu::Context::TypeVulkan) {
                struct {
                    unsigned int width;
                    unsigned int height;
                } params = { width, height };
                    vk_aplusb_matrix_bad.exec(params, workSize, a_gpu, b_gpu, c_gpu);
            } else {
            }

            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        double median_time = stats::median(times);
        unsigned long long data_size = (unsigned long long) 3 * width * height * sizeof(unsigned int);
        double bandwidth_gb_s = data_size / median_time / (1024*1024*1024);
        std::cout << "a + b matrix BAD kernel median VRAM bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
            

        std::vector<unsigned int> cs(width * height, 0);
        c_gpu.read(cs.data(), cs.size() * sizeof(unsigned int));
        for (size_t i = 0; i < width * height; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
    }

    {
        std::cout << "Running GOOD matrix kernel..." << std::endl;

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            gpu::WorkSize workSize(16, 16, width, height);


            if (context.type() == gpu::Context::TypeOpenCL) {
                ocl_aplusb_matrix_good.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
            } else if (context.type() == gpu::Context::TypeCUDA) {
                #if defined(CUDA_SUPPORT)
                    cuda::aplusb_matrix_good(workSize, a_gpu, b_gpu, c_gpu, width, height);
                #endif              
            } else if (context.type() == gpu::Context::TypeVulkan) {
                struct {
                    unsigned int width;
                    unsigned int height;
                } params = { width, height };
                    vk_aplusb_matrix_good.exec(params, workSize, a_gpu, b_gpu, c_gpu);
            } else {
                
            }

            times.push_back(t.elapsed());
        }
        std::cout << "a + b matrix kernel times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        double median_time = stats::median(times);
        unsigned long long data_size = (unsigned long long) 3 * width * height * sizeof(unsigned int);
        double bandwidth_gb_s = data_size / median_time / (1024*1024*1024);
        std::cout << "a + b matrix GOOD kernel median VRAM bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    

        std::vector<unsigned int> cs(width * height, 0);
        c_gpu.read(cs.data(), cs.size() * sizeof(unsigned int));
        for (size_t i = 0; i < width * height; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
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
            return 0;
        } if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            return 0;
        } else {
            return 1;
        }
    }

    return 0;
}
