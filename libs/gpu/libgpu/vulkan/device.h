#pragma once

#include "../../../base/libbase/data_type.h"

#include <set>
#include <vector>
#include <string>

namespace vk {
	enum class Format;
	namespace raii {
		class Instance;
	}
}

namespace avk2 {
	class Device {
	public:
		Device(size_t vk_device_index);

		bool										init(bool silent=true);
		bool										init(const vk::raii::Instance &instance, bool silent=true);

		bool										supportsFreeMemoryRequest();

		bool										supportsExtension(const std::string &extension_name);

		std::vector<DataType>						supportedImageDataTypes() const;
		vk::Format									typeToVkDepthStencilFormat(DataType type) const;
		vk::Format									typeToVkFormat(DataType type) const;
		vk::Format									typeToVkFormat(DataType type, size_t nchannels) const;

		size_t										freeMemory();
		size_t										freeMemory(const vk::raii::Instance &instance);

		void										printInfo();

		size_t										device_id_vulkan;

		std::string									name;
		unsigned int								device_type; // vk::PhysicalDeviceType
		unsigned int								vendor_id;
		std::string									vendor_name;
		std::string									api_version;
		std::string									driver_version;
		unsigned long long							mem_size; // in bytes
		unsigned int								max_workgroup_size;

		unsigned int								min_storage_buffer_offset_alignment;
		unsigned int								max_fragment_output_attachments;
		unsigned int								max_image_array_layers;
		unsigned int								max_image_dimension_2d;

		unsigned int								pci_domain;
		unsigned int								pci_bus;
		unsigned int								pci_device;
		unsigned int								pci_function;

		std::set<std::string>						extensions;
	};
}
