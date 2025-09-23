#include <gtest/gtest.h>

#include "device.h"

#include <libbase/gtest_utils.h>

class DeviceInfo {
public:
	DeviceInfo()
	{
		// this is minimal list of relevant properties - only they are respected in mergeApisOnDevices()
		name_ = "no name";

		// note that in practice these values are always specified for NVIDIA and often specified for AMD,
		// but for test-case description simplicity (and for strictness) - let's suppose there are no PCI-E info
		pci_bus_id_ = 0;
		pci_device_id_ = 0;

		supports_vulkan_ = false;
		supports_opencl_ = false;
		supports_cuda_ = false;

		// note that std::sort is also used inside mergeApisOnDevices(), and it relies on 'operator< (const Device &other)',
		// which also takes into account OpenCL/Vulkan device handles
		// in practice they are different for different devices, but for test-case description simplicity - let's suppose they are constant
		device_id_vulkan_ = 1;
		device_id_opencl_ = (cl_device_id) 2;
	}

	DeviceInfo &setPCI(unsigned int pci_bus_id, unsigned int pci_device_id)
	{
		pci_bus_id_ = pci_bus_id;
		pci_device_id_ = pci_device_id;
		return *this;
	}

	DeviceInfo &enableCUDA()
	{
		supports_cuda_ = true;
		return *this;
	}

	DeviceInfo &enableOpenCL()
	{
		supports_opencl_ = true;
		return *this;
	}

	DeviceInfo &enableVulkan()
	{
		supports_vulkan_ = true;
		return *this;
	}

	gpu::Device createDevice()
	{
		gpu::Device device;
		device.name = name_;
		device.pci_bus_id = pci_bus_id_;
		device.pci_device_id = pci_device_id_;
		device.supports_vulkan = supports_vulkan_;
		device.supports_opencl = supports_opencl_;
		device.supports_cuda = supports_cuda_;
		device.device_id_vulkan = device_id_vulkan_;
		device.device_id_opencl = device_id_opencl_;
		return device;
	}

	std::string		name_;
	uint32_t		pci_bus_id_;
	uint32_t		pci_device_id_;
	bool			supports_vulkan_;
	bool			supports_opencl_;
	bool			supports_cuda_;
	uint64_t		device_id_vulkan_;
	cl_device_id	device_id_opencl_;
};

#define NAME_NV_RTX_3080		std::string("DEVICE NVIDIA RTX 3080")
#define NAME_AMD_V520			std::string("DEVICE AMD V520")
#define NAME_INTEL_ARC580		std::string("DEVICE INTEL ARC A580")

#define EXPECTED_THE_SAME_DEVICES_LIST {}

DeviceInfo createCudaDevice(std::string name)
{
	DeviceInfo info;
	info.name_ = name;
	info.supports_cuda_ = true;
	return info;
}

DeviceInfo createOpenCLDevice(std::string name)
{
	DeviceInfo info;
	info.name_ = name;
	info.supports_opencl_ = true;
	return info;
}

DeviceInfo createVulkanDevice(std::string name)
{
	DeviceInfo info;
	info.name_ = name;
	info.supports_vulkan_ = true;
	return info;
}

void checkMergeTestCase(std::vector<DeviceInfo> initial_devices, std::vector<DeviceInfo> expected_merged_devices)
{
	if (expected_merged_devices.size() == 0 && std::vector<DeviceInfo>(EXPECTED_THE_SAME_DEVICES_LIST).size() == 0) {
		expected_merged_devices = initial_devices;
	}

	std::vector<gpu::Device> devices;
	for (auto d: initial_devices) {
		devices.push_back(d.createDevice());
	}

	gpu::mergeApisOnDevices(devices);

	EXPECT_EQ(devices.size(), expected_merged_devices.size());
	for (size_t i = 0; i < std::min(devices.size(), expected_merged_devices.size()); ++i) {
		gpu::Device expected_device = expected_merged_devices[i].createDevice();
		EXPECT_EQ(devices[i], expected_device);
	}
}

TEST(gpu_device, mergeApisOnDevices)
{
	if (isDebuggerAttached()) {
		std::cout << "attached debugger detected" << std::endl;
		gtest::forceBreakOnFailure();
	}

	// CUDA -> CUDA
	checkMergeTestCase({createCudaDevice(NAME_NV_RTX_3080)},
					   EXPECTED_THE_SAME_DEVICES_LIST);
	// CUDA, CUDA -> CUDA, CUDA
	checkMergeTestCase({createCudaDevice(NAME_NV_RTX_3080), createCudaDevice(NAME_NV_RTX_3080)},
					   EXPECTED_THE_SAME_DEVICES_LIST);
	// CUDA, CUDA, CUDA -> CUDA, CUDA, CUDA
	checkMergeTestCase({createCudaDevice(NAME_NV_RTX_3080), createCudaDevice(NAME_NV_RTX_3080), createCudaDevice(NAME_NV_RTX_3080)},
					   EXPECTED_THE_SAME_DEVICES_LIST);

	// OpenCL -> OpenCL
	checkMergeTestCase({createOpenCLDevice(NAME_AMD_V520)},
					   EXPECTED_THE_SAME_DEVICES_LIST);
	// OpenCL, OpenCL -> OpenCL, OpenCL		- describes real-world use-case from ticket #5206 (PCI-E=0/0 for AMD V520 on windows VMs)
	checkMergeTestCase({createOpenCLDevice(NAME_AMD_V520), createOpenCLDevice(NAME_AMD_V520)},
					   EXPECTED_THE_SAME_DEVICES_LIST);
	// OpenCL, OpenCL, OpenCL -> OpenCL, OpenCL, OpenCL
	checkMergeTestCase({createOpenCLDevice(NAME_AMD_V520), createOpenCLDevice(NAME_AMD_V520), createOpenCLDevice(NAME_AMD_V520)},
					   EXPECTED_THE_SAME_DEVICES_LIST);

	// CUDA, OpenCL -> CUDA + OpenCL
	checkMergeTestCase({createCudaDevice(NAME_NV_RTX_3080), createOpenCLDevice(NAME_NV_RTX_3080)},
					   {createCudaDevice(NAME_NV_RTX_3080).enableOpenCL()});

	// OpenCL, CUDA -> CUDA + OpenCL
	checkMergeTestCase({createOpenCLDevice(NAME_NV_RTX_3080), createCudaDevice(NAME_NV_RTX_3080)},
					   {createCudaDevice(NAME_NV_RTX_3080).enableOpenCL()});

	// CUDA, OpenCL -> CUDA + OpenCL if devices have the same PCI-E bus/device id
	checkMergeTestCase({createCudaDevice(NAME_NV_RTX_3080).setPCI(1, 2), createOpenCLDevice(NAME_NV_RTX_3080).setPCI(1, 2)},
					   {createCudaDevice(NAME_NV_RTX_3080).setPCI(1, 2).enableOpenCL()});

	// CUDA, OpenCL -> CUDA, OpenCL if devices have different PCI-E bus/device id
	checkMergeTestCase({createCudaDevice(NAME_NV_RTX_3080).setPCI(1, 2), createOpenCLDevice(NAME_NV_RTX_3080).setPCI(3, 4)},
					   EXPECTED_THE_SAME_DEVICES_LIST);
}