#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>

#ifdef _MSC_VER
int setenv(const char *name, const char *value, int overwrite);
#else
#include <stdlib.h> // for setenv
#endif


namespace vk {
	struct PhysicalDeviceMemoryProperties;
	struct QueueFamilyProperties;
	struct ApplicationInfo;
}


namespace avk2 {
	std::optional<unsigned int> getIndexOfDeviceLocalHeap(const vk::PhysicalDeviceMemoryProperties& device_memory_properties);

	std::optional<unsigned int> getIndexOfQueueFamily(const std::vector<vk::QueueFamilyProperties>& queues_family_properties);

	std::string decodeAPIVersion(uint32_t apiVersion);

	std::string decodeVendorID(uint32_t vendorID);

	std::string decodeDriverVersion(uint32_t driverVersion, uint32_t vendorID);

	void reportVmaAssert(bool condition, int line);

	void reportError(ptrdiff_t err, size_t unique_code, int line, const std::string &prefix = std::string());
}

#define VK_CHECK_RESULT(f, unique_code)																	\
{																										\
	std::string str(#f);																				\
	bool enable_debug_mode = false;																		\
	if (enable_debug_mode)																				\
		std::cerr << str + "\n" << std::flush;															\
	VkResult res = VkResult(f);																			\
	avk2::reportError(res, unique_code, __LINE__, str + ": ");											\
}

#define VULKAN_TIMEOUT_NANOSECS		(1000*1000*1000) // 1 second (in nanosecs)
//#define VULKAN_NO_TIMEOUT			std::numeric_limits<uint64_t>::max()

#define STAGING_BUFFER_SIZE			(8*1024*1024) // 8 MB
