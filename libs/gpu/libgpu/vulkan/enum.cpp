#include "enum.h"

#include "libbase/timer.h"
#include <libgpu/device.h>
#include <libbase/runtime_assert.h>
#include <iostream>

#include "engine.h"
#include "vulkan_api_headers.h"

avk2::VulkanEnum::VulkanEnum()
{
}

avk2::VulkanEnum::~VulkanEnum()
{
}

bool avk2::VulkanEnum::compareDevice(const Device &dev1, const Device &dev2)
{
	if (dev1.name							> dev2.name)							return false;
	if (dev1.name							< dev2.name)							return true;
	if (dev1.device_id_vulkan				> dev2.device_id_vulkan)				return false;
	return true;
}

bool avk2::VulkanEnum::enumDevices(bool silent)
{
	try {
		VKF.init();

		auto instance_context = avk2::InstanceContext::getGlobalInstanceContext();

		std::vector<vk::raii::PhysicalDevice> devices = instance_context->instance().enumeratePhysicalDevices();
		for (int device_index = 0; device_index < devices.size(); ++device_index) {
			Device device(device_index);
			if (!device.init(silent)) {
				continue;
			}

			rassert(device.name == devices[device_index].getProperties().deviceName, 32431247812943);

			devices_.push_back(device);
		}

		std::sort(devices_.begin(), devices_.end(), compareDevice);
		return true;
	} catch (vk::Error& e) {
		if (!silent) std::cerr << "Vulkan devices enumeration failed: " << e.what() << std::endl;
		return false;
	} catch (std::runtime_error& e) {
		// Note that call to "VKF.init(...)" can raise error "Failed to load vulkan library!".
		// It should be an InitializationFailedError (which is a vk::Error handled above),
		// but in fact in vulkan.hpp it is a std::runtime_error - https://github.com/KhronosGroup/Vulkan-Hpp/blob/2fbc146feefa43b8201af4b01eb3570110f9fa32/vulkan/vulkan.hpp#L16721-L16722
		// because:
		// > "there should be an InitializationFailedError, but msvc insists on the symbol does not exist within the scope of this function."
		if (!silent) std::cerr << "Vulkan devices enumeration failed: " << e.what() << std::endl;
		return false;
	}
}
