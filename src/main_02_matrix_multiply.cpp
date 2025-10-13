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

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_matrix03MultiplyNaive(ocl::getMatrix03MultiplyNaive());
    ocl::KernelSource ocl_matrix04MultiplyViaLocalMemory(ocl::getMatrix04MultiplyViaLocalMemory());

    avk2::KernelSource vk_matrix03MultiplyNaive(avk2::getMatrix03MultiplyNaive());
    avk2::KernelSource vk_matrix04MultiplyViaLocalMemory(avk2::getMatrix04MultiplyViaLocalMemory());
    avk2::KernelSource vk_matrix05MultiplyCooperativeMatrix(avk2::getMatrix05MultiplyCooperativeMatrix());

    unsigned int ksize = 64;
    unsigned int w = ksize * 32;
    unsigned int k = ksize * 8;
    unsigned int h = ksize * 16;
    std::cout << "c = a x b, matrices size: c (rows=h=" << h << " * cols=w=" << w << ")"
              << " = a (rows=h=" << h << " x cols=k=" << k << ") * b (rows=k=" << k << " x cols=w=" << w << ")" << std::endl;
    std::cout << "matrices data size: a - " << sizeof(float) * h * k / 1024 / 1024 << " mb, b - " << sizeof(float) * k * w / 1024 / 1024 << " mb, c - " << sizeof(float) * k * w / 1024 / 1024 << " mb" << std::endl;

    std::vector<float> input_a_cpu(h * k, 0);  // rows=h * cols=k
    std::vector<float> input_b_cpu(k * w, 0);  // rows=k * cols=w
    std::vector<float> output_c_cpu(h * w, 0); // rows=h * cols=w
    std::vector<float> output_c_gpu(h * w, 0); // rows=h * cols=w
    FastRandom r;

    for (size_t i = 0; i < input_a_cpu.size(); ++i) {

        input_a_cpu[i] = r.nextf();
    }
    for (size_t i = 0; i < input_b_cpu.size(); ++i) {
        input_b_cpu[i] = r.nextf();
    }

    // allocate buf to vram
    gpu::gpu_mem_32f matrix_a_gpu(h * k); // rows=h * cols=k
    gpu::gpu_mem_32f matrix_b_gpu(k * w); // rows=k * cols=w
    gpu::gpu_mem_32f matrix_c_gpu(h * w); // rows=h * cols=w

    // load input via pci-e bus: cpu ram -> gpu vram
    matrix_a_gpu.writeN(input_a_cpu.data(), h * k);
    matrix_b_gpu.writeN(input_b_cpu.data(), h * k);

    std::vector<std::string> algorithm_names = {
        "cpu_w_OpenMP",
        "01_naive",
        "02_w_lmem"
    };

    bool I_Want_Super_Puper_Prestige_Points = false; // =;(((

    for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {

        const std::string& algorithm = algorithm_names[algorithm_index];
        std::cout << "______________________________________________________" << std::endl;
        std::cout << "evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

        std::vector<double> times;
        int iters_count = (algorithm == "cpu_w_OpenMP") ? 1 : 10; // cpu is too slow
        for (int iter = 0; iter < iters_count; ++iter) {
		
            timer t;

            if (algorithm == "cpu_w_OpenMP") {

                cpu::multiply(input_a_cpu, input_b_cpu, output_c_cpu, w, h, k, true);
            } else {
								   //
                if (context.type() == gpu::Context::TypeOpenCL) {

                    if (algorithm == "01_naive") {

                        ocl_matrix03MultiplyNaive.exec(gpu::WorkSize(1, 1, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);

                    } else if (algorithm == "02_w_lmem") {

                        ocl_matrix04MultiplyViaLocalMemory.exec(gpu::WorkSize(16, 16, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
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

	// compute eff algo throughput
        double total_ops = 1.0 * h * w * (k + k - 1); // n additions and multipclications
        double gflops = 1000*1000*1000;
        std::cout << "algorithm gflops: " << total_ops / gflops / stats::median(times) << " gflops" << std::endl;
        std::cout << "algorithm effective memory bandwidth: " << 1.0 * (h * k + k * w + h * w) * sizeof(float) / 1024 / 1024 / 1024 / stats::median(times) << " gbps" << std::endl;

        if (algorithm != "cpu_w_OpenMP") {

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
            std::cout << "relative differences with cpu: " << stats::valuesStatsLine(relative_errors) << std::endl;

            float median_relative_error = stats::median(relative_errors);
            float perc99_relative_error = stats::percentile(relative_errors, 99);

            std::cout << "median relative difference with cpu: " << median_relative_error << std::endl;
            std::cout << "99% percentile relative difference with cpu: " << perc99_relative_error << std::endl;
	    
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
	    
        std::cerr << "error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {

	    return 0;
        }
        if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
   	
	    return 0;
        } else {
    	
	    return 1;
        }
    }

    return 0;
}
