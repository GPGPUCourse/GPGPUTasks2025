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

    ocl::KernelSource ocl_sum01Atomics(ocl::getSum01Atomics());
    ocl::KernelSource ocl_sum02AtomicsLoadK(ocl::getSum02AtomicsLoadK());
    ocl::KernelSource ocl_sum03LocalMemoryAtomicPerWorkgroup(ocl::getSum03LocalMemoryAtomicPerWorkgroup());
    ocl::KernelSource ocl_sum04LocalReduction(ocl::getSum04LocalReduction());

    unsigned int n = 100 * 1000 * 1000;
    rassert(n % LOAD_K_VALUES_PER_ITEM == 0, 4356345432524); // for simplicity
    std::vector<unsigned int> values(n, 0);
    size_t cpu_sum = 0;
    for (size_t i = 0; i < n; ++i) {

        values[i] = (3 * (i + 5) + 7) % 17;
        cpu_sum += values[i];
        rassert(cpu_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, cpu_sum, values[i], i); // ensure no overflow
    }

    // allocate buf to vram
    gpu::gpu_mem_32u input_gpu(n);
    gpu::gpu_mem_32u sum_accum_gpu(1);
    gpu::gpu_mem_32u reduction_buffer1_gpu(div_ceil(n, (unsigned int)GROUP_SIZE));
    gpu::gpu_mem_32u reduction_buffer2_gpu(div_ceil(n, (unsigned int)GROUP_SIZE));
    
    std::vector<double> pci_times;

    for (size_t iter = 0; iter < 10; ++iter) {
	    
	    timer t;
	    
	    input_gpu.writeN(values.data(), n);

	    pci_times.push_back(t.elapsed());
    }

    double memory_sent_size_gb = sizeof(unsigned int) * n / 1024.0 / 1024.0 / 1024.0; // same as later
    std::cout << "pci-e eff throughput: " << memory_sent_size_gb / stats::median(pci_times) << " gb/s" << std::endl;

    std::vector<std::string> algorithm_names = {
        "cpu",
        "cpu_w_OpenMP",
        "01_atomicAdd_f_each_workItem",
        "02_atomicAdd_b_each_workItem_loads_K_values",
        "03_local_memory_&_atomicAdd_f_master_thread",
        "04_local_reduction",
    };

    for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {
	    
        const std::string& algorithm = algorithm_names[algorithm_index];
        std::cout << "______________________________________________________" << std::endl;
        std::cout << "evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

        std::vector<double> times;
        unsigned int gpu_sum = 0;
        for (int iter = 0; iter < 10; ++iter) {
		
            timer t;

            if (algorithm == "cpu") {

                gpu_sum = cpu::sum(values.data(), n);
            } else if (algorithm == "cpu_w_OpenMP") {

                gpu_sum = cpu::sumOpenMP(values.data(), n);
            } else {
		    
	        if (context.type() == gpu::Context::TypeOpenCL) {
			
                    if (algorithm == "01_atomicAdd_f_each_workItem") {
			    
                        sum_accum_gpu.fill(0);
                        ocl_sum01Atomics.exec(gpu::WorkSize(GROUP_SIZE, n), input_gpu, sum_accum_gpu, n);
                        sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "02_atomicAdd_b_each_workItem_loads_K_values") {

                        sum_accum_gpu.fill(0);
                        ocl_sum02AtomicsLoadK.exec(gpu::WorkSize(GROUP_SIZE, n / LOAD_K_VALUES_PER_ITEM), input_gpu, sum_accum_gpu, n);
                        sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "03_local_memory_&_atomicAdd_f_master_thread") {

			sum_accum_gpu.fill(0);
                        ocl_sum03LocalMemoryAtomicPerWorkgroup.exec(gpu::WorkSize(GROUP_SIZE, n), input_gpu, sum_accum_gpu, n);
			sum_accum_gpu.readN(&gpu_sum, 1);
                    } else if (algorithm == "04_local_reduction") {

			unsigned int num_groups = div_ceil(n, (unsigned int)GROUP_SIZE);
			
			reduction_buffer1_gpu.fill(0);
			ocl_sum04LocalReduction.exec(gpu::WorkSize(GROUP_SIZE, n), input_gpu, reduction_buffer1_gpu, n);
			
			std::vector<unsigned int> partials = reduction_buffer1_gpu.readVector();
			gpu_sum = 0;
			
			for (unsigned int i = 0; i < num_groups; ++i) {
				
				gpu_sum += partials[i];
			}                    
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

        double memory_size_gb = sizeof(unsigned int) * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "sum median effective algorithm bandwidth: " << memory_size_gb / stats::median(times) << " gb/s" << std::endl;

        rassert(cpu_sum == gpu_sum, 3452341235234456, cpu_sum, gpu_sum);

        // check whether input remains intact
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
