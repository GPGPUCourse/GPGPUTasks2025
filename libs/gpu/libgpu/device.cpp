#include "device.h"
#include "context.h"

#include <libgpu/opencl/enum.h>
#include <libgpu/vulkan/enum.h>
#include <libbase/string_utils.h>
#include <libbase/runtime_assert.h>
#include <libbase/timer.h>
#include <algorithm>

#ifdef CUDA_SUPPORT
#include <libgpu/cuda/enum.h>
#include <libgpu/cuda/utils.h>
#include <cuda_runtime.h>
#endif

namespace gpu {

bool Device::enable_cuda = true;

std::vector<Device> enumDevices(bool cuda_silent, bool opencl_silent, bool vk_silent)
{
	std::vector<Device> devices;

	timer tm_total, tm_cuda, tm_opencl, tm_vulkan;
	tm_total.start();

#ifdef CUDA_SUPPORT
	bool cuda_device_found = false;
	if (Device::enable_cuda) {
		tm_cuda.start();

		CUDAEnum cuda_enum;
		cuda_enum.enumDevices(cuda_silent);

		const std::vector<CUDAEnum::Device> &cuda_devices = cuda_enum.devices();
		cuda_device_found = (cuda_devices.size() > 0);
		for (size_t k = 0; k < cuda_devices.size(); k++) {
			const CUDAEnum::Device &cuda_device = cuda_devices[k];

			Device device;
			device.name						= cuda_device.name;
			device.plain_name				= cuda_device.name;
			device.compute_units			= cuda_device.compute_units;
			device.clock					= cuda_device.clock;
			device.mem_size					= cuda_device.mem_size;
			device.pci_bus_id				= cuda_device.pci_bus_id;
			device.pci_device_id			= cuda_device.pci_device_id;
			device.cuda_compcap_major		= cuda_device.compcap_major;
			device.cuda_compcap_minor		= cuda_device.compcap_minor;
			device.opencl_device_type		= CL_DEVICE_TYPE_GPU;
			device.vendor_id				= VENDOR::ID_NVIDIA;
			device.vendor_name				= "Nvidia Corporation";
			device.supports_cuda			= true;
			device.device_id_cuda			= cuda_device.id;
			devices.push_back(device);
		}

		tm_cuda.stop();
	}
#endif

	{
		tm_opencl.start();

		OpenCLEnum opencl_enum;
		opencl_enum.enumDevices(opencl_silent);

		const std::vector<OpenCLEnum::Device> &opencl_devices = opencl_enum.devices();
		for (size_t k = 0; k < opencl_devices.size(); k++) {
			const OpenCLEnum::Device &opencl_device = opencl_devices[k];

//#ifdef CUDA_SUPPORT
//			if (cuda_device_found && (opencl_device.vendor_id == VENDOR::ID_NVIDIA
//									  || opencl_device.vendor == "NVIDIA Corporation"
//									  || opencl_device.vendor == "NVIDIA"))
//				continue;
//#endif

			Device device;
			device.name						= opencl_device.name;
			device.plain_name				= opencl_device.plain_name;
			device.opencl_device_type		= opencl_device.device_type;
			device.vendor_id				= opencl_device.vendor_id;
			device.vendor_name				= opencl_device.vendor;
			device.opencl_version			= opencl_device.version;
			device.opencl_driver_version	= opencl_device.driver_version;
			device.opencl_unified_memory	= opencl_device.unified_memory;
			device.compute_units			= opencl_device.compute_units;
			device.clock					= opencl_device.clock;
			device.mem_size					= opencl_device.mem_size;
			device.supports_opencl			= true;
			device.device_id_opencl			= opencl_device.id;

			if (opencl_device.vendor_id == VENDOR::ID_NVIDIA) {
				device.pci_bus_id			= opencl_device.nvidia_pci_bus_id;
				device.pci_device_id		= opencl_device.nvidia_pci_slot_id;
			} else if (opencl_device.vendor_id == VENDOR::ID_AMD) {
				device.pci_bus_id			= opencl_device.device_topology_amd.pcie.bus;
				device.pci_device_id		= opencl_device.device_topology_amd.pcie.device;
			} else {
				device.pci_bus_id			= 0;
				device.pci_device_id		= 0;
			}
			devices.push_back(device);
		}

		tm_opencl.stop();
	}

	{
		tm_vulkan.start();

		avk2::VulkanEnum vulkan_enum;
		vulkan_enum.enumDevices(vk_silent);

		const std::vector<avk2::Device> &vulkan_devices = vulkan_enum.devices();
		for (size_t k = 0; k < vulkan_devices.size(); k++) {
			const avk2::Device &vulkan_device = vulkan_devices[k];

			Device device;
			device.name						= vulkan_device.name;
			device.plain_name				= vulkan_device.name;
			device.opencl_device_type		= vulkan_device.device_type;
			device.vendor_id				= vulkan_device.vendor_id;
			device.vendor_name				= vulkan_device.vendor_name;
			device.vulkan_api_version		= vulkan_device.api_version;
			device.vulkan_driver_version	= vulkan_device.driver_version;
			device.compute_units			= 0;
			device.clock					= 0;
			device.mem_size					= vulkan_device.mem_size;
			device.supports_vulkan			= true;
			device.device_id_vulkan			= vulkan_device.device_id_vulkan;
			device.pci_bus_id				= vulkan_device.pci_bus;
			device.pci_device_id			= vulkan_device.pci_device;

			devices.push_back(device);
		}

		tm_vulkan.stop();
	}

	mergeApisOnDevices(devices);

//#ifdef CUDA_SUPPORT
//	// remove OpenCL option for CUDA GPUs
//	for (size_t k = 0; k < devices.size(); k++) {
//		if (devices[k].supports_cuda)
//			devices[k].supports_opencl = false;
//	}
//#endif

	std::vector<std::string> details;
#ifdef CUDA_SUPPORT
	details.push_back("CUDA: " + to_string(tm_cuda.elapsed()) + " sec");
#endif
	details.push_back("OpenCL: " + to_string(tm_opencl.elapsed()) + " sec");
	details.push_back("Vulkan: " + to_string(tm_vulkan.elapsed()) + " sec");

	std::cout << "Found " << devices.size() << " GPUs in " << tm_total.elapsed() << " sec";
	if (details.size())
		std::cout << " (" + join(details, ", ") + ")";
	std::cout << std::endl;

	return devices;
}

void mergeApisOnDevices(std::vector<Device> &devices)
{
	std::sort(devices.begin(), devices.end());

	// merge corresponding devices
	for (size_t k = 0; k + 1 < devices.size(); k++) {
		// if (#K) differs from the next (#K+1) - they can't be merged, moving forward
		if (devices[k].name				!= devices[k + 1].name)				continue;
		if (devices[k].pci_bus_id		!= devices[k + 1].pci_bus_id)		continue;
		if (devices[k].pci_device_id	!= devices[k + 1].pci_device_id)	continue;

		// Order of APIs (thanks to sorting): CUDA, OpenCL, Vulkan
		if (!devices[k].supports_opencl && devices[k + 1].supports_opencl) { // If (#K) is CUDA and the next (#K+1) is OpenCL - we assimilate it into (#K)
			//rassert(devices[k].supports_cuda, 934512557887969); // typically - only CUDA device can be before OpenCL device, BUT it seems that GPU duplicates happens - see ticket #5174
			//rassert(!devices[k].supports_opencl, 567276361128344); // let's ensure that we didn't merge any OpenCL device yet
			rassert(!devices[k + 1].supports_cuda || !devices[k + 1].supports_vulkan, 351324124312412); // the next devices should be Vulkan-only (wasn't merged with any other device)
			devices[k].supports_opencl			= true;
			devices[k].device_id_opencl			= devices[k + 1].device_id_opencl;
			devices[k].opencl_driver_version	= devices[k + 1].opencl_driver_version;
			devices[k].opencl_unified_memory	= devices[k + 1].opencl_unified_memory;
			devices.erase(devices.begin() + k + 1);
			--k; // let's look at (#K) again - may be we can assimilate another one
		} else if (!devices[k].supports_vulkan && devices[k + 1].supports_vulkan) { // If (#K) is CUDA/OpenCL and the next (#K+1) is Vulkan - we assimilate it into (#K)
			rassert(devices[k].supports_cuda || devices[k].supports_opencl, 567276361128347); // only CUDA/OpenCL device can be before Vulkan device
			rassert(!devices[k].supports_vulkan, 567276361128345); // let's ensure that we didn't merge any Vulkan device yet
			rassert(!devices[k + 1].supports_cuda || !devices[k + 1].supports_opencl, 351324124312412);  // the next devices should be Vulkan-only (wasn't merged with any other device)
			devices[k].supports_vulkan			= true;
			devices[k].device_id_vulkan			= devices[k + 1].device_id_vulkan;
			devices[k].vulkan_api_version		= devices[k + 1].vulkan_api_version;
			devices[k].vulkan_driver_version	= devices[k + 1].vulkan_driver_version;
			devices.erase(devices.begin() + k + 1);
			--k; // let's look at (#K) again - may be we can assimilate another one
		} else {
			// we are from the same API, i.e. we are guaranteed by that API to be different devices
			rassert((devices[k].supports_opencl && devices[k + 1].supports_opencl)
					|| (devices[k].supports_cuda && devices[k + 1].supports_cuda)
					|| (devices[k].supports_vulkan && devices[k + 1].supports_vulkan), 7844723252466102);
		}
	}
}

Device::Device()
{
	compute_units		= 0;
	clock				= 0;
	mem_size			= 0;
	pci_bus_id			= 0;
	pci_device_id		= 0;
	vendor_id			= 0;
	opencl_device_type	= 0;
	opencl_unified_memory= false;
	cuda_compcap_major	= 0;
	cuda_compcap_minor	= 0;

	supports_opencl		= false;
	supports_cuda		= false;
	supports_vulkan		= false;

	device_id_opencl	= 0;
	device_id_cuda		= 0;
	device_id_vulkan	= 0;
}

bool Device::isCPU() const
{
	return opencl_device_type == CL_DEVICE_TYPE_CPU;
}

bool Device::isGPU() const
{
	return opencl_device_type == CL_DEVICE_TYPE_GPU;
}

bool Device::isIntel() const
{
	return (vendor_id == VENDOR::ID_INTEL || tolower(vendor_name).find("intel") != std::string::npos);
}

bool Device::isAmd() const
{
	return (vendor_id == VENDOR::ID_AMD || tolower(vendor_name).find("advanced micro devices") != std::string::npos);
}

bool Device::isNvidia() const
{
	return (vendor_id == VENDOR::ID_NVIDIA || tolower(vendor_name).find("nvidia") != std::string::npos);
}

bool Device::isApple() const
{
	return (vendor_id == VENDOR::ID_APPLE || tolower(vendor_name).find("apple") != std::string::npos);
}

bool Device::printInfo() const
{
#ifdef CUDA_SUPPORT
	if (supports_cuda) {
		return CUDAEnum::printInfo(device_id_cuda, opencl_driver_version);
	}
#endif

	if (supports_opencl) {
		return OpenCLEnum::printInfo(device_id_opencl);
	}

	if (supports_vulkan) {
		avk2::Device vk_device(device_id_vulkan);
		rassert(vk_device.init(true), 606279660904311);
		vk_device.printInfo();
		return true;
	}

	return false;
}

bool Device::supportsFreeMemoryQuery() const
{
#ifdef CUDA_SUPPORT
	if (supports_cuda) {
		return true;
	} else
#endif
	if (supports_opencl) {
		ocl::DeviceInfo device_info;
		device_info.init(device_id_opencl);
		if (device_info.hasExtension(CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXT)) {
			return true;
		}
	}
	if (supports_vulkan) {
		avk2::Device vk_device(device_id_vulkan);
		vk_device.init();
		if (vk_device.supportsFreeMemoryRequest()) {
			return true;
		}
	}

	return false;
}

uint64_t Device::getFreeMemory() const
{
#ifdef CUDA_SUPPORT
	if (supports_cuda) {
		Context context;
		if (!context.isActive()) {
			context.init(device_id_cuda);
			context.activate();
		}

		size_t total_mem_size = 0;
		size_t free_mem_size = 0;
		CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_size, &total_mem_size));
		return free_mem_size;
	}
#endif
	if (supports_opencl) {
		ocl::DeviceInfo device_info;
		device_info.init(device_id_opencl);
		if (device_info.supportsFreeMemoryRequest()) {
			return device_info.freeMemory();
		}
	}
	if (supports_vulkan) {
		avk2::Device vk_device(device_id_vulkan);
		vk_device.init();
		if (vk_device.supportsFreeMemoryRequest()) {
			return vk_device.freeMemory();
		}
	}
	uint64_t free_mem_size = mem_size - mem_size / 5;
	return free_mem_size;
}

unsigned int defaultMask(const std::vector<Device> &devices)
{
	unsigned int mask = 0;

	for (size_t k = 0; k < devices.size(); k++) {
		const Device &device = devices[k];

		if (!device.isGPU())
			continue;
		bool device_is_discrete_AMD = device.isAmd() && !device.opencl_unified_memory; // AMD APUs are not enabled by default 
		bool whitelisted = device_is_discrete_AMD || device.isNvidia() || device.isApple();
		if (!whitelisted)
			continue;

		mask |= (1 << k);
	}

	return mask;
}

std::vector<Device>	selectAllDevices(unsigned int mask, bool cuda_silent, bool opencl_silent, bool vk_silent)
{
	if (!mask)
		return std::vector<Device>();

	std::vector<Device> devices = enumDevices(cuda_silent, opencl_silent, vk_silent);

	std::vector<Device> res;
	for (size_t k = 0; k < devices.size(); k++) {
		if (!(mask & (1 << k)))
			continue;

		const Device &device = devices[k];

		if ((device.supports_opencl && !opencl_silent) || (device.supports_cuda && !cuda_silent) || (device.supports_vulkan && !vk_silent))
			if (!device.printInfo())
				continue;

		res.push_back(device);
	}

	return res;
}

std::vector<Device> selectAllDevices(unsigned int mask, bool silent)
{
	return selectAllDevices(mask, silent, silent, silent);
}

std::vector<Device> selectComputeDevices(unsigned int mask, bool opencl_cuda_silent)
{
	std::vector<Device> devices = selectAllDevices(mask, opencl_cuda_silent, opencl_cuda_silent, true);

	std::vector<Device> res;
	for (size_t k = 0; k < devices.size(); k++) {
		if (devices[k].supports_opencl || devices[k].supports_cuda) {
			res.push_back(devices[k]);
		}
	}

	return res;
}

std::vector<Device> selectVulkanDevices(unsigned int mask, bool vk_silent)
{
	std::vector<Device> devices = selectAllDevices(mask, true, true, vk_silent);

	std::vector<Device> res;
	for (size_t k = 0; k < devices.size(); k++) {
		if (devices[k].supports_vulkan) {
			res.push_back(devices[k]);
		}
	}

	return res;
}

}
