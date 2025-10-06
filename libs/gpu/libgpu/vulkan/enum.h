#pragma once

#include "device.h"

#include <set>
#include <string>
#include <vector>
#include <memory>

namespace avk2 {
	class VulkanEnum {
	public:
		VulkanEnum();
		~VulkanEnum();

		bool	enumDevices(bool silent);
		std::vector<Device> &	devices()	{ return devices_;		}

	protected:
		static	bool	compareDevice(const Device &dev1, const Device &dev2);

		std::vector<Device>		devices_;
	};
}