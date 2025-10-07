#include <vector>
#include <cmath>
#include <numeric>

#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libutils/misc.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/opencl/engine.h>
#include "kernels/kernels.h"

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = gpu::activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_partition_scan(ocl::getPrefixScan32(), "partition_scan");
    ocl::KernelSource ocl_global_scan(ocl::getPrefixScan32(), "global_scan");
    ocl::KernelSource ocl_prefix_scan64(ocl::getPrefixScan64(), "prefix_scan");

    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    unsigned int total_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        as[i] = unsigned((3 * (i + 5) + 7) % 17);
        total_sum += as[i];
    }

    const unsigned int matrixHeight = 64; // must match MATRIX_HEIGHT define
    const unsigned int tile = 896;        // must match TILE define
    const unsigned int lcm_th = std::lcm(tile, matrixHeight);
    const unsigned int tiled_n = ((n + lcm_th - 1) / lcm_th) * lcm_th;
    const unsigned int tiles = tiled_n / tile;

    std::vector<unsigned int> padded(tiled_n, 0);
    std::copy(as.begin(), as.end(), padded.begin());

    gpu::gpu_mem_32u data_gpu(tiled_n);
    data_gpu.writeN(padded.data(), tiled_n);

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        if (context.type() == gpu::Context::TypeOpenCL) {
            data_gpu.writeN(padded.data(), tiled_n);
            const unsigned int partitions_total = tiles * matrixHeight;
            gpu::gpu_mem_32u global_counter_gpu(1);
            std::vector<unsigned int> zero_counter(1, 0);
            global_counter_gpu.writeN(zero_counter.data(), 1);

            gpu::gpu_mem_64f states_gpu(partitions_total);
            std::vector<double> zeros64(partitions_total, 0.0);
            states_gpu.writeN(zeros64.data(), zeros64.size());

            gpu::gpu_mem_32u data_out_gpu(tiled_n);
            gpu::WorkSize ws(matrixHeight, partitions_total);
            ocl_prefix_scan64.exec(ws, data_gpu, data_out_gpu, global_counter_gpu, states_gpu);
            data_out_gpu.copyToN(data_gpu, tiled_n);
        } else if (context.type() == gpu::Context::TypeCUDA) {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
        } else if (context.type() == gpu::Context::TypeVulkan) {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
        } else {
            rassert(false, 4531412341, context.type());
        }

        times.push_back(t.elapsed());
    }
    std::cout << "prefix scan times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "prefix scan median effective VRAM bandwidth: " << memory_size_gb / stats::median(times) << " GB/s" << std::endl;

    std::vector<unsigned int> gpu_prefix_sum = data_gpu.readVector(tiled_n);

    unsigned int cpu_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        cpu_sum += as[i];
        rassert(cpu_sum == gpu_prefix_sum[i], 566324523452323, cpu_sum, gpu_prefix_sum[i], i);
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
