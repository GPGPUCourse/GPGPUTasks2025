#include <gtest/gtest.h>

#include <libbase/timer.h>
#include <libbase/stats.h>
#include <libbase/fast_random.h>

#include "test_utils.h"

#include "kernels/defines.h"
#include "kernels/kernels.h"

TEST(vulkan, binarySearch)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		avk2::KernelSource kernel_batched_binary_search(avk2::getBatchedBinarySearch());

		int max_value = 10000000;
		int n_searches = max_value + 1;
		std::vector<int> n_values_to_test = {10, 100, 1000, 1000000};
		int niters = 3;

		for (int n_values: n_values_to_test) {
			std::cout << "setup: " << n_searches << " searches x [0 ... " << max_value << "] values in sorted array of " << n_values << " size" << std::endl;

			FastRandom r(239);

			std::vector<int> values_to_find(n_searches);
			std::vector<int> sorted_values(n_values);
			std::vector<int> result_indices_expected(n_searches);
			for (size_t i = 0; i < n_values; ++i) {
				sorted_values[i] = r.next(0, max_value);
			}
			std::sort(sorted_values.begin(), sorted_values.end());
			for (size_t i = 0; i < n_searches; ++i) {
				int value = i;
				values_to_find[i] = value;
				auto it = std::lower_bound(sorted_values.begin(), sorted_values.end(), value);
				if (it == sorted_values.end() || *it != value) {
					// NOT FOUND
				}
				result_indices_expected[i] = it - sorted_values.begin();
			}

			gpu::gpu_mem_32i values_to_find_gpu(n_searches);
			gpu::gpu_mem_32i sorted_values_gpu(n_values);
			gpu::gpu_mem_32i result_indices_gpu(n_searches);

			values_to_find_gpu.writeN(values_to_find.data(), n_searches);
			sorted_values_gpu.writeN(sorted_values.data(), n_values);

			std::vector<double> iter_times;
			for (int iter = 0; iter < niters; ++iter) {
				timer timer; timer.start();
				kernel_batched_binary_search.exec(std::pair(n_searches, n_values), gpu::WorkSize(VK_GROUP_SIZE, n_searches),
												  values_to_find_gpu, sorted_values_gpu, result_indices_gpu);
				iter_times.push_back(timer.elapsed());

				std::vector<int> result_indices_found = result_indices_gpu.readVector();
				for (size_t i = 0; i < n_searches; ++i) {
					EXPECT_EQ(result_indices_found[i], result_indices_expected[i]);
				}
			}
			std::cout << "kernel launches times: " << stats::valuesStatsLine(iter_times) << std::endl;
		}
	}

	checkPostInvariants();
}
