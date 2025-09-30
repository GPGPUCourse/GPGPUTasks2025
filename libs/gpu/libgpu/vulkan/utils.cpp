#include "utils.h"

#include <libgpu/utils.h>
#include <libbase/runtime_assert.h>

#include "vulkan_api_headers.h"

#include "exceptions.h"


#ifdef _MSC_VER
// see https://stackoverflow.com/a/23616164
int setenv(const char *name, const char *value, int overwrite)
{
	int errcode = 0;
	if(!overwrite) {
		size_t envsize = 0;
		errcode = getenv_s(&envsize, NULL, 0, name);
		if(errcode || envsize) return errcode;
	}
	return _putenv_s(name, value);
}
#endif

std::optional<unsigned int> avk2::getIndexOfDeviceLocalHeap(const vk::PhysicalDeviceMemoryProperties& device_memory_properties)
{
	for (unsigned int type_index = 0; type_index < device_memory_properties.memoryTypeCount; ++type_index) {
		// searching for the first memory type that has DeviceLocal-bit flag (and has the same flag in its memory heap)
		// TODO add support for APU devices like AMD device in ticket #5124 - it seems to have only 256 MB of device-local VRAM,
		//  so probably in cases when size<VK_GPU_MIN_VRAM_REQUIRED we should choose bigger heap (relaxing the device-local requirement)
		vk::MemoryPropertyFlags type_flags = device_memory_properties.memoryTypes[type_index].propertyFlags;
		unsigned int heap_index = device_memory_properties.memoryTypes[type_index].heapIndex;
		rassert(heap_index < device_memory_properties.memoryHeapCount, 3247182937219312);
		vk::MemoryHeapFlags heap_flags = device_memory_properties.memoryHeaps[heap_index].flags;
		if ((type_flags & vk::MemoryPropertyFlagBits::eDeviceLocal) && (heap_flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
			return heap_index;
		}
	}
	return {};
}

std::optional<unsigned int> avk2::getIndexOfQueueFamily(const std::vector<vk::QueueFamilyProperties>& queues_family_properties)
{
	for (unsigned int queue_index = 0; queue_index < queues_family_properties.size(); ++queue_index) {
		const vk::QueueFamilyProperties &properties = queues_family_properties[queue_index];
		if ((properties.queueFlags & vk::QueueFlagBits::eGraphics) || (properties.queueFlags & vk::QueueFlagBits::eCompute)) {
			return queue_index;
		}
	}
	return {};
}

// see https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/RAII_Samples/PhysicalDeviceProperties/PhysicalDeviceProperties.cpp
std::string avk2::decodeAPIVersion(uint32_t apiVersion)
{
	return std::to_string(VK_VERSION_MAJOR(apiVersion)) + "." + std::to_string(VK_VERSION_MINOR(apiVersion)) + "." +
		   std::to_string(VK_VERSION_PATCH(apiVersion));
}

std::string avk2::decodeVendorID(uint32_t vendorID)
{
	// below 0x10000 are the PCI vendor IDs (https://pcisig.com/membership/member-companies)
	if ( vendorID < 0x10000 )
	{
			 if (vendorID == gpu::VENDOR::ID_AMD)	{ return "Advanced Micro Devices Inc."; }
		else if (vendorID == gpu::VENDOR::ID_NVIDIA) { return "Nvidia Corporation";		  }
		else if (vendorID == gpu::VENDOR::ID_INTEL)  { return "Intel Corporation";		   }
		else if (vendorID == gpu::VENDOR::ID_APPLE)  { return "Apple Inc.";				  }
		else return "VendorID#" + std::to_string(vendorID);
	} else {
	  // above 0x10000 should be vkVendorIDs
	  return vk::to_string(vk::VendorId(vendorID));
	}
}

// see https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/RAII_Samples/PhysicalDeviceProperties/PhysicalDeviceProperties.cpp
std::string avk2::decodeDriverVersion(uint32_t driverVersion, uint32_t vendorID)
{
	if (vendorID == gpu::VENDOR::ID_NVIDIA) {
		return std::to_string((driverVersion >> 22) & 0x3FF) + "." + std::to_string((driverVersion >> 14) & 0xFF) + "."
			   + std::to_string((driverVersion >> 6) & 0xFF) + "." + std::to_string(driverVersion & 0x3F);
	} else if (vendorID == gpu::VENDOR::ID_INTEL) {
		return std::to_string( ( driverVersion >> 14 ) & 0x3FFFF ) + "." + std::to_string( ( driverVersion & 0x3FFF ) );
	} else {
		return decodeAPIVersion(driverVersion);
	}
}

void avk2::reportVmaAssert(bool condition, int line)
{
	if (condition)
		return;

	std::cerr << "assertion in VMA failed at line " << line << std::endl;
}

void avk2::reportError(ptrdiff_t err, size_t unique_code, int line, const std::string &prefix)
{
	if (VK_SUCCESS == err)
		return;

	std::string message = prefix + vk::to_string((vk::Result) err) + " (" + to_string(err) + ") with code " + to_string(unique_code) + " at line " + to_string(line);

	switch (err) {
		case VK_ERROR_OUT_OF_DEVICE_MEMORY:
			throw vk_bad_alloc(message);
		default:
			throw vk_exception(message);
	}
}