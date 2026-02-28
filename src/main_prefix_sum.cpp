#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <memory>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_fill_with_zeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_sum_reduction(ocl::getPrefixSum01Reduction());
    ocl::KernelSource ocl_prefix_accumulation(ocl::getPrefixSum02PrefixAccumulation());

    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    size_t total_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        as[i] = (3 * (i + 5) + 7) % 17;
        total_sum += as[i];
        rassert(total_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, total_sum, as[i], i);
    }

    gpu::gpu_mem_32u input_gpu(n), prefix_sum_accum_gpu(n);

    std::vector<std::shared_ptr<gpu::gpu_mem_32u>> block_sums;
    std::vector<unsigned int> level_sizes;
    unsigned int cur_size = n;
    level_sizes.push_back(cur_size);
    while (cur_size > 1) {
        unsigned int next_size = div_ceil(cur_size, (unsigned int)GROUP_SIZE);
        block_sums.push_back(std::make_shared<gpu::gpu_mem_32u>(next_size));
        level_sizes.push_back(next_size);
        cur_size = next_size;
    }

    input_gpu.writeN(as.data(), n);

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) {
        timer t;

        if (context.type() == gpu::Context::TypeOpenCL) {
            for (size_t i = 0; i < block_sums.size(); ++i) {
                unsigned int size = level_sizes[i];
                gpu::WorkSize ws(GROUP_SIZE, div_ceil(size, (unsigned int)GROUP_SIZE) * GROUP_SIZE);
                
                if (i == 0) {
                    ocl_sum_reduction.exec(ws, input_gpu, prefix_sum_accum_gpu, *block_sums[0], size);
                } else {
                    ocl_sum_reduction.exec(ws, *block_sums[i - 1], *block_sums[i - 1], *block_sums[i], size);
                }
            }

            for (int i = (int)block_sums.size() - 2; i >= 0; --i) {
                unsigned int size = level_sizes[i];
                gpu::WorkSize ws(GROUP_SIZE, div_ceil(size, (unsigned int)GROUP_SIZE) * GROUP_SIZE);
                
                if (i == 0) {
                    ocl_prefix_accumulation.exec(ws, prefix_sum_accum_gpu, *block_sums[0], size);
                } else {
                    ocl_prefix_accumulation.exec(ws, *block_sums[i - 1], *block_sums[i], size);
                }
            }
        } else {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
        }

        times.push_back(t.elapsed());
    }
    std::cout << "prefix sum time, s " << stats::valuesStatsLine(times) << std::endl;

    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "prefix sum median effective VRAM bandwidth, gb/s " << memory_size_gb / stats::median(times) << std::endl;

    std::vector<unsigned int> gpu_prefix_sum = prefix_sum_accum_gpu.readVector();

    size_t cpu_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        cpu_sum += as[i];
        rassert(cpu_sum == gpu_prefix_sum[i], 566324523452323, cpu_sum, gpu_prefix_sum[i], i);
    }

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
        std::string err_msg = e.what();
        if (err_msg == std::string(DEVICE_NOT_SUPPORT_API) || err_msg == std::string(CODE_IS_NOT_IMPLEMENTED)) {
            return 0;
        } else {
            return 1;
        }
    }
    return 0;
}