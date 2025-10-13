#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <iomanip>

void run(int argc, char** argv) {

    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeOpenCL);

    ocl::KernelSource ocl_matrix01TransposeNaive(ocl::getMatrix01TransposeNaive());
    ocl::KernelSource ocl_matrix02TransposeCoalescedViaLocalMemory(ocl::getMatrix02TransposeCoalescedViaLocalMemory());

    avk2::KernelSource vk_matrix01TransposeNaive(avk2::getMatrix01TransposeNaive());
    avk2::KernelSource vk_matrix02TransposeCoalescedViaLocalMemory(avk2::getMatrix02TransposeCoalescedViaLocalMemory());

    unsigned int ksize = 256;
    unsigned int w = ksize * 32;
    unsigned int h = ksize * 16;
    std::cout << "matrix size: rows=h=" << h << " x cols=w=" << w << " (" << sizeof(float) * w * h / 1024 / 1024 << " mb)" << std::endl;

    std::vector<float> input_cpu(h * w, 0);
    FastRandom r;
    for (size_t i = 0; i < h * w; ++i) {

        input_cpu[i] = r.nextf();
    }

    // allocate to vram
    gpu::gpu_mem_32f input_matrix_gpu (h * w); // rows=h * cols=w
    gpu::gpu_mem_32f output_matrix_gpu(w * h); // rows=w * cols=h

    // load input via pci-e bus: cpu ram -> gpu vram
    input_matrix_gpu.writeN(input_cpu.data(), h * w);

    std::vector<std::string> algorithm_names = {

        "01_naive_transpose",
        "02_transpose_v_lmem"
    };

    for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {

        const std::string& algorithm = algorithm_names[algorithm_index];

        std::cout << "______________________________________________________" << std::endl;
        std::cout << "evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

        std::vector<double> times;
        for (int iter = 0; iter < 10; ++iter) {

            timer t;

	    if (context.type() == gpu::Context::TypeOpenCL) {

                if (algorithm == "01_naive_transpose") {

                    ocl_matrix01TransposeNaive.exec(gpu::WorkSize(1, 1, w, h), input_matrix_gpu, output_matrix_gpu, w, h);
                } else if (algorithm == "02_transpose_v_lmem") {

                    ocl_matrix02TransposeCoalescedViaLocalMemory.exec(gpu::WorkSize(16, 16, w, h), input_matrix_gpu, output_matrix_gpu, w, h);
                } else {

                    rassert(false, 652345234321, algorithm, algorithm_index);
                }
            } else {

                rassert(false, 546345243, context.type());
            }

            times.push_back(t.elapsed());
        }
        std::cout << "algorithm times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

	// compute max eff algo throughput
        double memory_size_gb = 2.0 * sizeof(float) * w * h / 1024.0 / 1024.0 / 1024.0;
        std::cout << "median effective algorithm bandwidth: " << memory_size_gb / stats::median(times) << " gbps" << std::endl;

        std::vector<float> results = output_matrix_gpu.readVector(); // input matrix: w x h -> output matrix: h x w
        for (size_t j = 0; j < h; ++j) {

            for (size_t i = 0; i < w; ++i) {
		    
                rassert(results[i * h + j] == input_cpu[j * w + i], 6573452432, i, j);
            }
        }
    }
}

int main(int argc, char** argv) {

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
