#include <gtest/gtest.h>

#include "libbase/timer.h"
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "libbase/runtime_assert.h"

#include "kernels/kernels.h"

std::vector<gpu::Device> enumOpenCLDevices()
{
	gpu::Device::enable_cuda = false; // because otherwise enumDevices will ignore OpenCL support on NVIDIA devices
	std::vector<gpu::Device> devices = gpu::enumDevices(false, false, false);

	std::vector<gpu::Device> ocl_devices;
	for (auto &device : devices) {
		if (device.supports_opencl) {
			ocl_devices.push_back(device);
		}
	}
	std::cout << "OpenCL supported devices: " << ocl_devices.size() << " out of " << devices.size() << std::endl;

	return ocl_devices;
}

gpu::Context activateOpenCLContext(gpu::Device &device)
{
	rassert(device.supports_opencl, 354312345151);
	device.supports_cuda = false;
	device.supports_vulkan = false;

	rassert(device.printInfo(), 456134512341);

	gpu::Context gpu_context;
	gpu_context.init(device.device_id_opencl);
	gpu_context.activate();
	return gpu_context;
}

TEST(opencl, activateContexts)
{
	std::vector<gpu::Device> devices = enumOpenCLDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing OpenCL device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateOpenCLContext(devices[k]);
	}
}

TEST(opencl, writeRead)
{
	std::vector<gpu::Device> devices = enumOpenCLDevices();
	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing OpenCL device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateOpenCLContext(devices[k]);

		unsigned int n = 10 * 1000 * 1000;
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
}

TEST(opencl, aplusb)
{
	std::vector<gpu::Device> devices = enumOpenCLDevices();

	unsigned int n = 10 * 1000 * 1000;

	std::vector<unsigned int> as(n, 0);
	std::vector<unsigned int> bs(n, 0);
	std::vector<unsigned int> cs(n, 0);
	for (size_t i = 0; i < n; ++i) {
		as[i] = 3 * (i + 5) + 7;
		bs[i] = 11 * (i + 13) + 17;
	} 

	for (size_t k = 0; k < devices.size(); ++k) {
		std::cout << "Testing OpenCL device #" << (k + 1) << "/" << devices.size() << "..." << std::endl;
		gpu::Context context = activateOpenCLContext(devices[k]);

		gpu::gpu_mem_32u gpu_a, gpu_b, gpu_c;
		gpu_a.resizeN(n);
		gpu_b.resizeN(n);
		gpu_c.resizeN(n);

		gpu_a.writeN(as.data(), n);
		gpu_b.writeN(bs.data(), n);

		ocl::KernelSource kernel_aplusb(ocl::getAplusBKernel(), "aplusb");

		double rw_data_size = 3.0 * n * sizeof(as[0]);
		timer exec_timer;
		for (int iter = 0; iter < 3; ++iter) {
			exec_timer.restart();
			gpu::WorkSize worksize(128, n);
			kernel_aplusb.exec(worksize, gpu_a, gpu_b, gpu_c, n);
			std::cout << "VRAM bandwidth: " << (rw_data_size / exec_timer.elapsed()) / (1024 * 1024 * 1024) << " GB/s" << std::endl;

			gpu_c.readN(cs.data(), n);
			for (size_t i = 0; i < n; ++i) {
				rassert(as[i] + bs[i] == cs[i], 6478216416635168);
			}
		}
	}
}