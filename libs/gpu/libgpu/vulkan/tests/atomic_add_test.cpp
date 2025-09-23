#include <gtest/gtest.h>

#include <libbase/timer.h>
#include <libbase/stats.h>
#include <libbase/fast_random.h>

#include "test_utils.h"

#include "kernels/defines.h"
#include "kernels/kernels.h"

TEST(vulkan, atomicAdd)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		avk2::KernelSource kernel_atomic_add(avk2::getAtomicAddKernel());

		std::vector<int> setup_n_additions = {5,	10,		100000,		10000000,	10000000};
		std::vector<int> setup_table_sizes = {10,	5,		1000000,	100000000,	1000	};
		std::vector<int> setup_max_addings = {1024,	1024,	8,			1,			1		};

		int niters = 3;

		for (int setup = 0; setup < setup_n_additions.size(); ++setup) {
			int n_additions = setup_n_additions[setup];
			int table_size = setup_table_sizes[setup];
			int max_adding_value = setup_max_addings[setup];
			std::cout << "setup: " << n_additions << " additions x [0 ... " << max_adding_value << "] values into table of " << table_size << " size" << std::endl;

			int max_float_precise_int = (1 << std::numeric_limits<float>::digits) - 1;
			rassert(n_additions * max_adding_value <= max_float_precise_int, 215254685);

			FastRandom r(239);

			std::vector<int> indices_to_add(n_additions);
			std::vector<float> values_to_add(n_additions);
			std::vector<int> table_values_expected(table_size);
			for (size_t i = 0; i < n_additions; ++i) {
				int index = r.next(0, table_size - 1);
				int value = r.next(0, max_adding_value);
				indices_to_add[i] = index;
				values_to_add[i] = value;

				table_values_expected[index] += value;
			}

			gpu::gpu_mem_32i indicies_to_add_gpu(n_additions);
			gpu::gpu_mem_32f values_to_add_gpu(n_additions);
			gpu::gpu_mem_32f table_values_gpu(table_size);

			indicies_to_add_gpu.writeN(indices_to_add.data(), n_additions);
			values_to_add_gpu.writeN(values_to_add.data(), n_additions);

			std::vector<double> iter_times;
			for (int iter = 0; iter < niters; ++iter) {
				table_values_gpu.fill(0);

				timer timer; timer.start();
				kernel_atomic_add.exec(std::pair(n_additions, table_size), gpu::WorkSize(VK_GROUP_SIZE, n_additions),
									   indicies_to_add_gpu, values_to_add_gpu, table_values_gpu);
				iter_times.push_back(timer.elapsed());

				std::vector<float> table_values_found = table_values_gpu.readVector();
				for (size_t i = 0; i < table_size; ++i) {
					EXPECT_EQ(table_values_found[i], table_values_expected[i]);
				}
			}
			std::cout << "kernel launches times: " << stats::valuesStatsLine(iter_times) << std::endl;
		}
	}

	checkPostInvariants();
}
