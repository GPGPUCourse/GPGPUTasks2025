#include <gtest/gtest.h>

#include <libbase/timer.h>
#include <libbase/omp_utils.h>
#include <libgpu/shared_device_buffer.h>

#include "test_utils.h"

#include "kernels/kernels.h"
#include "kernels/defines.h"

TEST(vulkan, activateContexts)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);
	}

	checkPostInvariants();
}

TEST(vulkan, activateContextsMultiThreaded)
{
	OMP_DISPATCHER_INIT
	#pragma omp parallel for schedule(dynamic, 1)
	for (ptrdiff_t iter = 0; iter < 16; ++iter) {
		OMP_TRY
		bool silent = (iter != 0);
		std::vector<gpu::Device> devices = enumVKDevices(silent);
		for (size_t k = 0; k < devices.size(); ++k) {
			gpu::Context context = activateVKContext(devices[k], silent);
		}
		OMP_CATCH
	}
	OMP_RETHROW

	checkPostInvariants();
}

TEST(vulkan, writeRead)
{
	std::vector<gpu::Device> devices = enumVKDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		unsigned int n = 100 * 1000 * 1000;
		std::vector<unsigned int> as(n, 0);
		std::vector<unsigned int> bs(n, 0);
		for (size_t i = 0; i < n; ++i) {
			as[i] = 3 * (i + 5) + 7;
		}

		gpu::gpu_mem_32u gpu_a;
		gpu_a.resizeN(n);

		double data_size = n * sizeof(as[0]);
		timer pcie_timer;
		for (int iter = 0; iter < 3; ++iter) {
			pcie_timer.restart();
			gpu_a.writeN(as.data(), n);
			std::cout << "RAM --PCI-E-> VRAM write bandwidth: " << (data_size / pcie_timer.elapsed()) / (1024 * 1024) << " MB/s" << std::endl;

			pcie_timer.restart();
			gpu_a.readN(bs.data(), n);
			std::cout << "RAM <-PCI-E-- VRAM  read bandwidth: " << (data_size / pcie_timer.elapsed()) / (1024 * 1024) << " MB/s" << std::endl;

			for (size_t i = 0; i < n; ++i) {
				rassert(as[i] == bs[i], 45123451243214312);
			}
		}
	}

	checkPostInvariants();
}

TEST(vulkan, aplusb)
{
	std::vector<gpu::Device> devices = enumVKDevices();

	unsigned int n = 100 * 1000 * 1000;

	std::vector<unsigned int> as(n, 0);
	std::vector<unsigned int> bs(n, 0);
	std::vector<unsigned int> cs(n, 0);
	for (size_t i = 0; i < n; ++i) {
		as[i] = 3 * (i + 5) + 7;
		bs[i] = 11 * (i + 13) + 17;
	} 

	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing Vulkan device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateVKContext(devices[k]);

		gpu::gpu_mem_32u gpu_a, gpu_b, gpu_c;
		gpu_a.resizeN(n);
		gpu_b.resizeN(n);
		gpu_c.resizeN(n);

		gpu_a.writeN(as.data(), n);
		gpu_b.writeN(bs.data(), n);

		avk2::KernelSource kernel_aplusb(avk2::getAplusBKernel());

		double rw_data_size = 3.0 * n * sizeof(as[0]);
		timer exec_timer;
		for (int launch = 0; launch < 3; ++launch) {
			exec_timer.restart();
			gpu::WorkSize worksize(VK_GROUP_SIZE, n);
			kernel_aplusb.exec(n, worksize, gpu_a, gpu_b, gpu_c);
			std::cout << "VRAM bandwidth: " << (rw_data_size / exec_timer.elapsed()) / (1024 * 1024 * 1024) << " GB/s" << std::endl;

			gpu_c.readN(cs.data(), n);
			for (size_t i = 0; i < n; ++i) {
				rassert(as[i] + bs[i] == cs[i], 6478216416635168);
			}
		}

		std::vector<unsigned int> wrong_bs = bs;
		wrong_bs[n / 2] += 1;
		gpu_b.writeN(wrong_bs.data(), n);
		gpu::WorkSize worksize(VK_GROUP_SIZE, n);
		// we expect that GPU rassert 160682133 will fail:
		EXPECT_THROW(kernel_aplusb.exec(n, worksize, gpu_a, gpu_b, gpu_c), gpu_failure);
	}

	checkPostInvariants();
}

TEST(vulkan, aplusbMultiThreaded)
{
	unsigned int n = 10 * 1000 * 1000;

	OMP_DISPATCHER_INIT
	#pragma omp parallel
	{
		std::vector<unsigned int> as(n, 0);
		std::vector<unsigned int> bs(n, 0);
		std::vector<unsigned int> cs(n, 0);
		for (size_t i = 0; i < n; ++i) {
			as[i] = 3 * (i + 5) + 7;
			bs[i] = 11 * (i + 13) + 17;
		}

		#pragma omp for schedule(dynamic, 1)
		for (ptrdiff_t iter = 0; iter < 16; ++iter) {
			OMP_TRY
			bool silent = (iter != 0);
			std::vector<gpu::Device> devices = enumVKDevices(silent);
			for (size_t k = 0; k < devices.size(); ++k) {
				gpu::Context context = activateVKContext(devices[k], silent);

				gpu::gpu_mem_32u gpu_a, gpu_b, gpu_c;
				gpu_a.resizeN(n);
				gpu_b.resizeN(n);
				gpu_c.resizeN(n);

				gpu_a.writeN(as.data(), n);
				gpu_b.writeN(bs.data(), n);

				avk2::KernelSource kernel_aplusb(avk2::getAplusBKernel());

				for (int launch = 0; launch < 3; ++launch) {
					gpu::WorkSize worksize(VK_GROUP_SIZE, n);
					kernel_aplusb.exec(n, worksize, gpu_a, gpu_b, gpu_c);
				}

				gpu_c.readN(cs.data(), n);
				for (size_t i = 0; i < n; ++i) {
					rassert(as[i] + bs[i] == cs[i], 6478216416635168);
				}
			}
			OMP_CATCH
		}
	}
	OMP_RETHROW

	checkPostInvariants();
}
