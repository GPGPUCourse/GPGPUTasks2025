#include <libbase/stats.h>
#include <libutils/misc.h>
#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>
#include "kernels/defines.h"
#include "kernels/kernels.h"
#include "debug.h" 
#include <fstream>
#include <memory>

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);
    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_radixSort01LocalCounting(ocl::getRadixSort01LocalCounting());
    ocl::KernelSource ocl_radixSort02GlobalPrefixesScanSumReduction(ocl::getRadixSort02GlobalPrefixesScanSumReduction());
    ocl::KernelSource ocl_radixSort03GlobalPrefixesScanAccumulation(ocl::getRadixSort03GlobalPrefixesScanAccumulation());
    ocl::KernelSource ocl_radixSort04Scatter(ocl::getRadixSort04Scatter());

    FastRandom r;

    int n = 100*1000*1000;
    int max_value = std::numeric_limits<int>::max(); 
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> sorted(n, 0);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.next(0, max_value);
    }
    std::cout << "n=" << n << " max_value=" << max_value << std::endl;

    {
        size_t force_duplicates_attempts = 3;
        bool all_attempts_missed = true;
        for (size_t k = 0; k < force_duplicates_attempts; ++k) {
            size_t i = r.next(0, n - 1);
            size_t j = r.next(0, n - 1);
            if (i != j) {
                as[j] = as[i];
                all_attempts_missed = false;
            }
        }
        rassert(!all_attempts_missed, 4353245123412);
    }

    {
        sorted = as;
        std::cout << "sorting on CPU..." << std::endl;
        timer t;
        std::sort(sorted.begin(), sorted.end());
        double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU std::sort finished in " << t.elapsed() << " sec" << std::endl;
        std::cout << "CPU std::sort effective RAM bandwidth: " << memory_size_gb / t.elapsed() << " GB/s (" << n / 1000.0 / 1000.0 / t.elapsed() << " uint millions/s)" << std::endl;
    }

    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u buffer1_gpu(n); 
    gpu::gpu_mem_32u buffer_output_gpu(n);

    input_gpu.writeN(as.data(), n);

    unsigned int num_groups = div_ceil((unsigned int)n, (unsigned int)GROUP_SIZE);
    unsigned int counts_size = 16 * num_groups;
    gpu::gpu_mem_32u counts_gpu(counts_size);

    std::vector<std::shared_ptr<gpu::gpu_mem_32u>> block_sums;
    std::vector<unsigned int> level_sizes;
    unsigned int cur_size = counts_size;
    level_sizes.push_back(cur_size);
    while (cur_size > 1) {
        unsigned int next_size = div_ceil(cur_size, (unsigned int)GROUP_SIZE);
        block_sums.push_back(std::make_shared<gpu::gpu_mem_32u>(next_size));
        level_sizes.push_back(next_size);
        cur_size = next_size;
    }

    std::vector<double> times;
    for (int iter = 0; iter < 10; ++iter) { 
        timer t;

        if (context.type() == gpu::Context::TypeOpenCL) {
            gpu::gpu_mem_32u* src_buf = &input_gpu;
            gpu::gpu_mem_32u* dst_buf = &buffer1_gpu;

            // 8 проходов по 4 бита
            for (int pass = 0; pass < 8; ++pass) {
                unsigned int shift = pass * 4;

                ocl_radixSort01LocalCounting.exec(gpu::WorkSize(GROUP_SIZE, num_groups * GROUP_SIZE), 
                                                  *src_buf, counts_gpu, n, shift);

                for (size_t i = 0; i < block_sums.size(); ++i) {
                    unsigned int size = level_sizes[i];
                    gpu::WorkSize ws(GROUP_SIZE, div_ceil(size, (unsigned int)GROUP_SIZE) * GROUP_SIZE);
                    if (i == 0) {
                        ocl_radixSort02GlobalPrefixesScanSumReduction.exec(ws, counts_gpu, *block_sums[0], size);
                    } else {
                        ocl_radixSort02GlobalPrefixesScanSumReduction.exec(ws, *block_sums[i - 1], *block_sums[i], size);
                    }
                }

                for (int i = (int)block_sums.size() - 2; i >= 0; --i) {
                    unsigned int size = level_sizes[i];
                    gpu::WorkSize ws(GROUP_SIZE, div_ceil(size, (unsigned int)GROUP_SIZE) * GROUP_SIZE);
                    if (i == 0) {
                        ocl_radixSort03GlobalPrefixesScanAccumulation.exec(ws, counts_gpu, *block_sums[0], size);
                    } else {
                        ocl_radixSort03GlobalPrefixesScanAccumulation.exec(ws, *block_sums[i - 1], *block_sums[i], size);
                    }
                }

                ocl_radixSort04Scatter.exec(gpu::WorkSize(GROUP_SIZE, num_groups * GROUP_SIZE), 
                                            *src_buf, counts_gpu, *dst_buf, n, shift);
                if (pass == 0) {
                    src_buf = &buffer1_gpu;
                    dst_buf = &buffer_output_gpu;
                } else {
                    std::swap(src_buf, dst_buf);
                }
            }
        } else {
            throw std::runtime_error(CODE_IS_NOT_IMPLEMENTED);
        }

        times.push_back(t.elapsed());
    }
    std::cout << "GPU radix-sort times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU radix-sort median effective VRAM bandwidth, gb/s" << memory_size_gb / stats::median(times) << n / 1000.0 / 1000.0 / stats::median(times) << " uint millions/s" << std::endl;

    std::vector<unsigned int> gpu_sorted = buffer_output_gpu.readVector();
    for (size_t i = 0; i < n; ++i) {
        rassert(sorted[i] == gpu_sorted[i], 566324523452323, sorted[i], gpu_sorted[i], i);
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
        std::string err_msg = e.what();
        if (err_msg == std::string(DEVICE_NOT_SUPPORT_API) || err_msg == std::string(CODE_IS_NOT_IMPLEMENTED)) {
            return 0;
        } else {
            std::cerr << "Error: " << err_msg << std::endl;
            return 1;
        }
    }
    return 0;
}