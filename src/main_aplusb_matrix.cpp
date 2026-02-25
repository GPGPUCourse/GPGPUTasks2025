#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <iostream>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);
    ocl::KernelSource ocl_aplusb_matrix_bad(ocl::getAplusBMatrixBad());
    ocl::KernelSource ocl_aplusb_matrix_good(ocl::getAplusBMatrixGood());

    unsigned int task_size = 64;
    unsigned int width = task_size * 256;
    unsigned int height = task_size * 128;
    std::cout << "matrixes size: " << width << "x" << height << " = 3 * " 
              << (sizeof(unsigned int) * width * height / 1024 / 1024) << " MB" << std::endl;

    std::vector<unsigned int> as(width * height, 0);
    std::vector<unsigned int> bs(width * height, 0);
    for (size_t i = 0; i < width * height; ++i) {
        as[i] = 3 * (i + 5) + 7;
        bs[i] = 11 * (i + 13) + 17;
    }

    gpu::gpu_mem_32u a_gpu(width * height), b_gpu(width * height), c_gpu(width * height);
    a_gpu.writeN(as.data(), width * height);
    b_gpu.writeN(bs.data(), width * height);

    double mem_gb = (3.0 * width * height * sizeof(unsigned int)) / (1024.0 * 1024.0 * 1024.0);

    {
        std::cout << "\nRunning BAD matrix kernel..." << std::endl;
        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            gpu::WorkSize workSize(16, 16, width, height);
            
            if (context.type() == gpu::Context::TypeOpenCL) {
                ocl_aplusb_matrix_bad.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
            }

            times.push_back(t.elapsed());
        }
        std::cout << "BAD kernel times, s" << stats::valuesStatsLine(times) << std::endl;
        std::cout << "BAD kernel median bandwidth, gb/s" << mem_gb / stats::median(times) << std::endl;

        std::vector<unsigned int> cs(width * height, 0);
        c_gpu.readN(cs.data(), width * height);

        for (size_t i = 0; i < width * height; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230421312512, cs[i], as[i] + bs[i], i);
        }
    }

    {
        std::cout << "\nRunning GOOD matrix kernel..." << std::endl;
        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {
            timer t;

            gpu::WorkSize workSize(16, 16, width, height);
            
            if (context.type() == gpu::Context::TypeOpenCL) {
                ocl_aplusb_matrix_good.exec(workSize, a_gpu, b_gpu, c_gpu, width, height);
            }

            times.push_back(t.elapsed());
        }
        std::cout << "GOOD kernel times, s" << stats::valuesStatsLine(times) << std::endl;
        std::cout << "GOOD kernel median bandwidth, gb/s" << mem_gb / stats::median(times) << std::endl;

        std::vector<unsigned int> cs(width * height, 0);
        c_gpu.readN(cs.data(), width * height);

        for (size_t i = 0; i < width * height; ++i) {
            rassert(cs[i] == as[i] + bs[i], 321418230365731436, cs[i], as[i] + bs[i], i);
        }
    }
}

int main(int argc, char** argv)
{
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::string err_msg = e.what();
        if (err_msg == std::string(DEVICE_NOT_SUPPORT_API) || err_msg == std::string(CODE_IS_NOT_IMPLEMENTED)) {
            return 0;
        } else {
            return 1;
        }
    }
    return 0;
}