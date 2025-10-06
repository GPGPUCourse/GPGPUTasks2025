#pragma once

#include <string>
#include <vector>
#include <limits>
#include <cstdint>

typedef struct _cl_device_id *		cl_device_id;

namespace gpu {

class Device {
public:
	Device();

	std::string			name;
	std::string			plain_name;  // CL_DEVICE_NAME on nvidia & intel, CL_DEVICE_BOARD_NAME_AMD on amd
	uint64_t			opencl_device_type;
	uint32_t			vendor_id;
	std::string			vendor_name;
	std::string			opencl_version;
	std::string			opencl_driver_version;
	bool				opencl_unified_memory;
	uint32_t			cuda_compcap_major; // compute capability
	uint32_t			cuda_compcap_minor; // compute capability
	std::string			vulkan_api_version;
	std::string			vulkan_driver_version;
	uint32_t			compute_units;
	uint32_t			clock;
	uint64_t			mem_size;
	uint32_t			pci_bus_id;
	uint32_t			pci_device_id;

	bool				supports_opencl;
	bool				supports_cuda;
	bool				supports_vulkan;

	cl_device_id		device_id_opencl;
	int					device_id_cuda;
	uint64_t			device_id_vulkan;

	bool				isCPU() const;
	bool				isGPU() const;

	bool				isIntel() const;
	bool				isAmd() const;
	bool				isNvidia() const;
	bool				isApple() const;

	bool				printInfo() const;

	bool				supportsFreeMemoryQuery() const;
	uint64_t			getFreeMemory() const;

	bool operator< (const Device &other) const
	{
		if (name			< other.name)				return true;
		if (name			> other.name)				return false;
		if (pci_bus_id		< other.pci_bus_id)			return true;
		if (pci_bus_id		> other.pci_bus_id)			return false;
		if (pci_device_id	< other.pci_device_id)		return true;
		if (pci_device_id	> other.pci_device_id)		return false;
		// Order of APIs: CUDA, OpenCL, Vulkan
		if (supports_vulkan	< other.supports_vulkan)	return true;
		if (supports_vulkan	> other.supports_vulkan)	return false;
		if (supports_opencl	< other.supports_opencl)	return true;
		if (supports_opencl	> other.supports_opencl)	return false;
		if (supports_cuda	< other.supports_cuda)		return true;
		if (supports_cuda	> other.supports_cuda)		return false;
		if (device_id_vulkan< other.device_id_vulkan)	return true;
		if (device_id_vulkan> other.device_id_vulkan)	return false;
		if (device_id_opencl< other.device_id_opencl)	return true;
		if (device_id_opencl> other.device_id_opencl)	return false;
		return false;
	}

	bool operator == (const Device &other) const
	{
		if (name			!= other.name)				return false;
		if (pci_bus_id		!= other.pci_bus_id)		return false;
		if (pci_device_id	!= other.pci_device_id)		return false;
		if (supports_vulkan	!= other.supports_vulkan)	return false;
		if (supports_opencl	!= other.supports_opencl)	return false;
		if (supports_cuda	!= other.supports_cuda)		return false;
		if (device_id_vulkan!= other.device_id_vulkan)	return false;
		if (device_id_opencl!= other.device_id_opencl)	return false;
		return true;
	}

	static bool			enable_cuda;
};

// Enumerates CUDA/OpenCL/Vulkan GPGPU devices
std::vector<Device>	enumDevices(bool cuda_silent, bool opencl_silent, bool vk_silent);

// If on the same GPU device we have combination of available APIs (OpenCL/CUDA/Vulkan) - we will detect such GPU device
// multiple times - one time per each supported API, so we need to merge such duplicated device detections.
// To do so - we merge such duplicates but preserve supports_opencl/cuda/vulkan. 
void mergeApisOnDevices(std::vector<Device> &devices);

std::vector<Device>	selectAllDevices(unsigned int mask, bool cuda_silent, bool opencl_silent, bool vk_silent); // OpenCL/CUDA/Vulkan

#define ALL_GPUS std::numeric_limits<unsigned int>::max()

std::vector<Device>	selectAllDevices(unsigned int mask=ALL_GPUS, bool silent=false);     // OpenCL/CUDA/Vulkan
std::vector<Device>	selectComputeDevices(unsigned int mask=ALL_GPUS, bool opencl_cuda_silent=false); // OpenCL/CUDA
std::vector<Device>	selectVulkanDevices(unsigned int mask=ALL_GPUS, bool vk_silent=false);  // Vulkan

unsigned int		defaultMask(const std::vector<Device> &devices);

}
