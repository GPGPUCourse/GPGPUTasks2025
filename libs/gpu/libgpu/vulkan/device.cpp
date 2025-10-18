#include "device.h"

#include <libgpu/device.h>
#include <libbase/runtime_assert.h>

#include "vk/common_host.h"

#include "engine.h"
#include "utils.h"
#include "vulkan_api_headers.h"

#include <iostream>
#include <vector>

#define VK_CPU_DEVICES_ENABLED true

avk2::Device::Device(size_t device_id_vulkan)
: device_id_vulkan(device_id_vulkan)
{
	device_type			= 0;
	vendor_id			= 0;
	mem_size			= 0;

	pci_domain			= 0;
	pci_bus				= 0;
	pci_device			= 0;
	pci_function		= 0;
}

bool avk2::Device::init(bool silent)
{
	auto context_instance = avk2::InstanceContext::getGlobalInstanceContext();
	return init(context_instance->instance(), silent);
}

bool avk2::Device::init(const vk::raii::Instance &instance, bool silent)
{
	std::vector<vk::raii::PhysicalDevice> all_vk_devices = instance.enumeratePhysicalDevices();
	rassert(device_id_vulkan < all_vk_devices.size(), 2031530612780136, device_id_vulkan, all_vk_devices.size());
	vk::raii::PhysicalDevice vk_device = all_vk_devices[device_id_vulkan];

	vk::PhysicalDeviceProperties prop = vk_device.getProperties();

	if (!((prop.deviceType == vk::PhysicalDeviceType::eIntegratedGpu || prop.deviceType == vk::PhysicalDeviceType::eDiscreteGpu || prop.deviceType == vk::PhysicalDeviceType::eVirtualGpu)
		  || (VK_CPU_DEVICES_ENABLED && prop.deviceType == vk::PhysicalDeviceType::eCpu))) {
		// skpping because we are interested only in GPU (and, optionally, CPU)
		return false;
	}

	name								= vk::to_string(prop.deviceName);
	device_type							= (unsigned int) prop.deviceType;
	vendor_id							= prop.vendorID;
	vendor_name							= avk2::decodeVendorID(prop.vendorID);
	api_version							= avk2::decodeAPIVersion(prop.apiVersion);
	driver_version						= avk2::decodeDriverVersion(prop.driverVersion, prop.vendorID);
	max_workgroup_size					= prop.limits.maxComputeWorkGroupInvocations;
	if (max_workgroup_size < 256) {
		if (!silent) std::cout << "Vulkan device " << name << " skipped: too small max workgroup size " << max_workgroup_size << " (at least 256 required)" << std::endl;
		return false;
	}
	min_storage_buffer_offset_alignment	= prop.limits.minStorageBufferOffsetAlignment;
	if (GPU_BUFFER_SMALL_MAGIC_GUARD_NBYTES % min_storage_buffer_offset_alignment != 0 || GPU_BUFFER_BIG_MAGIC_GUARD_NBYTES % min_storage_buffer_offset_alignment != 0) {
		// to prevent Validation Error: [ VUID-VkWriteDescriptorSet-descriptorType-00328 ] | MessageID = 0xea08144e | vkUpdateDescriptorSets(): pDescriptorWrites[0].pBufferInfo[0].offset (4) must be a multiple of device limit minStorageBufferOffsetAlignment 16 when descriptor type is VK_DESCRIPTOR_TYPE_STORAGE_BUFFER. The Vulkan spec states: If descriptorType is VK_DESCRIPTOR_TYPE_STORAGE_BUFFER or VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, the offset member of each element of pBufferInfo must be a multiple of VkPhysicalDeviceLimits::minStorageBufferOffsetAlignment (https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkWriteDescriptorSet-descriptorType-00328)
		if (!silent) std::cout << "Vulkan device " << name << " skipped: unsupported min storage buffer offset alignment " << min_storage_buffer_offset_alignment << " (expected " << GPU_BUFFER_SMALL_MAGIC_GUARD_NBYTES << " or its divisor)" << std::endl;
		return false;
	}
	max_fragment_output_attachments		= prop.limits.maxFragmentOutputAttachments;
	if (max_fragment_output_attachments < VK_MAX_FRAGMENT_OUTPUT_ATTACHMENTS_USED) {
		// note that for simplicity we require support for the worst case, and it seems that max_fragment_output_attachments=8 nearly everywhere,
		// but it is possible to check supported value on per-task basis, respecting actual maximum channels number in dataset
		if (!silent) std::cout << "Vulkan device " << name << " skipped: too small max fragment output attachments " << max_fragment_output_attachments << " (at least " << VK_MAX_FRAGMENT_OUTPUT_ATTACHMENTS_USED << " required)" << std::endl;
		return false;
	}
	max_image_array_layers = prop.limits.maxImageArrayLayers;
	if (max_image_array_layers < VK_MAX_IMAGE_ARRAY_LAYERS_USED) {
		// note that for simplicity we require support for all tasks, and it seems that max_image_array_layers=2048 nearly everywhere,
		// but it is possible to check supported value on per-task basis
		if (!silent) std::cout << "Vulkan device " << name << " skipped: too small max image array layers " << max_image_array_layers << " (at least " << VK_MAX_IMAGE_ARRAY_LAYERS_USED << " required)" << std::endl;
		return false;
	}
	max_image_dimension_2d = prop.limits.maxImageDimension2D;
	if (prop.limits.maxComputeWorkGroupCount[0] < VK_MAX_COMPUTE_WORK_GROUP_COUNT_X || prop.limits.maxComputeWorkGroupCount[1] < VK_MAX_COMPUTE_WORK_GROUP_COUNT_Y || prop.limits.maxComputeWorkGroupCount[2] < VK_MAX_COMPUTE_WORK_GROUP_COUNT_Z) {
		if (!silent) std::cout << "Vulkan device " << name << " skipped: too small max compute workgroup count "
			<< prop.limits.maxComputeWorkGroupCount[0] << "x" << prop.limits.maxComputeWorkGroupCount[1] << "x" << prop.limits.maxComputeWorkGroupCount[1]
			<<" (at least " << VK_MAX_COMPUTE_WORK_GROUP_COUNT_X << "x" << VK_MAX_COMPUTE_WORK_GROUP_COUNT_Y << "x" << VK_MAX_COMPUTE_WORK_GROUP_COUNT_Z << " required)" << std::endl;
		return false;
	}

	if (VK_VERSION_MAJOR(prop.apiVersion) < VULKAN_MIN_VERSION_MAJOR
			|| (VK_VERSION_MAJOR(prop.apiVersion) == VULKAN_MIN_VERSION_MAJOR && VK_VERSION_MINOR(prop.apiVersion) < VULKAN_MIN_VERSION_MINOR)
			|| (VK_VERSION_MAJOR(prop.apiVersion) == VULKAN_MIN_VERSION_MAJOR && VK_VERSION_MINOR(prop.apiVersion) == VULKAN_MIN_VERSION_MINOR && VK_VERSION_PATCH(prop.apiVersion) < VULKAN_MIN_VERSION_PATCH)) {
		if (!silent) std::cout << "Vulkan device " << name << " skipped: at least " << VULKAN_MIN_VERSION_MAJOR << "." << VULKAN_MIN_VERSION_MINOR << "." << VULKAN_MIN_VERSION_PATCH << " Vulkan API version required (found: " << api_version << ")" << std::endl;
		return false;
	}

	vk::PhysicalDeviceMemoryProperties memory_properties = vk_device.getMemoryProperties();
	std::optional<unsigned int> device_local_heap_index = avk2::getIndexOfDeviceLocalHeap(memory_properties);
	if (device_local_heap_index.has_value()) {
		mem_size						= memory_properties.memoryHeaps[*device_local_heap_index].size;
	} else {
		if (!silent) std::cout << "Vulkan device " << name << " skipped: no device-local memory heap found" << std::endl;
		return false;
	}

	std::optional<unsigned int> queue_family_index = avk2::getIndexOfQueueFamily(vk_device.getQueueFamilyProperties());
	if (queue_family_index.has_value()) {
		// this is just for future - when we will create context we will create proper queue
	} else {
		if (!silent) std::cout << "Vulkan device " << name << " skipped: no queue with compute+graphics support found" << std::endl;
		return false;
	}

	auto device_features = vk_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceUniformBufferStandardLayoutFeatures>();
	vk::PhysicalDeviceUniformBufferStandardLayoutFeatures buffer_std_layout_features = *(vk::PhysicalDeviceUniformBufferStandardLayoutFeatures*) (device_features.get().pNext);
	rassert(buffer_std_layout_features.sType == buffer_std_layout_features.structureType, 822647462);
	// We require this feature to use std430 layout instead of std140
	if (!buffer_std_layout_features.uniformBufferStandardLayout) {
		if (!silent) std::cout << "Vulkan device " << name << " skipped: uniform buffer standard layout support is not found" << std::endl;
		return false;
	}

	if (avk2::isMoltenVK()) {
		// note that for now we need geometry shader ONLY for gl_PrimitiveID
		// but there is no geometry shader support on MoltenVK,
		// so we rely on fact that it seems that gl_PrimitiveID can be used on MoltenVK even without geometry shader feature requested
		// https://computergraphics.stackexchange.com/questions/9449/vulkan-using-gl-primitiveid-without-geometryshader-feature#comment14810_9449
	} else {
		if (!device_features.get().features.geometryShader) {
			if (!silent) std::cout << "Vulkan device " << name << " skipped: geometry shader support is not found" << std::endl; // requested for usage of gl_PrimitiveID
			return false;
		}
	}

	if (!device_features.get().features.fillModeNonSolid) {
		// note that we use wireframe rendering (see RenderBuilder::polygon_mode_wireframe_)
		// i.e. we use vk::PipelineRasterizationStateCreateInfo.polygonMode = vk::PolygonMode::eLine
		// to prevent Validation Error: [ VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-01507 ] | MessageID = 0x7a8a9b39 | vkCreateGraphicsPipelines(): pCreateInfos[0].pRasterizationState->polygonMode is VK_POLYGON_MODE_LINE, but fillModeNonSolid feature is note enabled. The Vulkan spec states: If the fillModeNonSolid feature is not enabled, polygonMode must be VK_POLYGON_MODE_FILL or VK_POLYGON_MODE_FILL_RECTANGLE_NV (https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-01507)
		if (!silent) std::cout << "Vulkan device " << name << " skipped: fillModeNonSolid support is not found" << std::endl; // requested for wireframe rendering, i.e.
		return false;
	}

	std::vector<vk::ExtensionProperties> extension_properties = vk_device.enumerateDeviceExtensionProperties();
	for (size_t k = 0; k < extension_properties.size(); ++k) {
		extensions.insert(extension_properties[k].extensionName);
	}

	if (extensions.count(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) {
		vk::PhysicalDevicePCIBusInfoPropertiesEXT pci_bus_info_properties = vk_device.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDevicePCIBusInfoPropertiesEXT>().get<vk::PhysicalDevicePCIBusInfoPropertiesEXT>();
		pci_domain					= pci_bus_info_properties.pciDomain;
		pci_bus						= pci_bus_info_properties.pciBus;
		pci_device					= pci_bus_info_properties.pciDevice;
		pci_function				= pci_bus_info_properties.pciFunction;
	} else {
		if (!silent) std::cout << "Vulkan device " << name << ": no pci bus info extension" << std::endl;
	}

	if (mem_size < VK_GPU_MIN_VRAM_REQUIRED) {
		if (!silent) std::cerr << "Vulkan device " << name << " skipped: too small memory size - " << (mem_size >> 20) << " MB" << std::endl;
		return false;
	}

	if (DEBUG_PRINTF_EXT_ENABLED && !extensions.count(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME)) {
		if (!silent) std::cerr << "Vulkan device " << name << " skipped: no shader non semantic info extension (required for debugPrintfEXT(...))" << std::endl;
		return false;
	}

	if (!extensions.count(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
		if (!silent) std::cerr << "Vulkan device " << name << " skipped: no VK_EXT_shader_atomic_float extension" << std::endl; // to use atomicAdd(float[], float), see support - https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_EXT_shader_atomic_float
		return false;
	}

	std::vector<vk::FormatFeatureFlagBits> generic_image_types_required_features = {
			vk::FormatFeatureFlagBits::eTransferSrc, vk::FormatFeatureFlagBits::eTransferDst, // check if we can read/write between RAM and VRAM into such image
			vk::FormatFeatureFlagBits::eSampledImage, // check if we can use such image with sampler (f.e. with bilinear interpolation)
	};

	std::vector<DataType> required_image_types = supportedImageDataTypes();
	for (DataType type: required_image_types) {
		vk::Format format = typeToVkFormat(type);
		vk::FormatProperties format_properties = vk_device.getFormatProperties(format);

		std::vector<vk::FormatFeatureFlagBits> feature_flags = generic_image_types_required_features;
		feature_flags.push_back(vk::FormatFeatureFlagBits::eStorageImage); // check if requested image format supports image storage operations required for storing pixel from the compute shader
		feature_flags.push_back(vk::FormatFeatureFlagBits::eColorAttachment); // check if we can output colors/pixel attributes from fragment shader into such image
		// no eDepthStencilAttachment because even eR32Sfloat doesn't support this feature - see https://vulkan.gpuinfo.org/displayreport.php?id=30317#formats and https://vulkan.gpuinfo.org/listoptimaltilingformats.php

		for (vk::FormatFeatureFlagBits feature_flag: feature_flags) {
			if (!(format_properties.optimalTilingFeatures & feature_flag)) {
				if (!silent) std::cout << "Vulkan device " << name << " skipped: optimalTilingFeatures support doesn't include " << vk::to_string(feature_flag) << " for " << vk::to_string(format) << std::endl;
				return false;
			}
		}
	}

	{
		vk::Format depth_format = typeToVkDepthStencilFormat(DataType32f);
		vk::FormatProperties format_properties = vk_device.getFormatProperties(depth_format);

		std::vector<vk::FormatFeatureFlagBits> depth_feature_flags = generic_image_types_required_features;
		depth_feature_flags.push_back(vk::FormatFeatureFlagBits::eDepthStencilAttachment); // check if such image can be used as depth framebuffer attachment
		// no eStorageImage and no eColorAttachment because eD32Sfloat doesn't support those features - see https://vulkan.gpuinfo.org/displayreport.php?id=30317#formats and https://vulkan.gpuinfo.org/listoptimaltilingformats.php

		for (vk::FormatFeatureFlagBits feature_flag: depth_feature_flags) {
			if (!(format_properties.optimalTilingFeatures & feature_flag)) {
				if (!silent) std::cout << "Vulkan device " << name << " skipped: optimalTilingFeatures support doesn't include " << vk::to_string(feature_flag) << " for " << vk::to_string(depth_format) << std::endl;
				return false;
			}
		}
	}

	return true;
}

bool avk2::Device::supportsFreeMemoryRequest()
{
	return supportsExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
}

bool avk2::Device::supportsExtension(const std::string &extension_name) const
{
	return extensions.count(extension_name) > 0;
}

std::vector<std::tuple<DataType, DataType, size_t, size_t, size_t>> avk2::Device::supportedCooperativeMatrixSizes() const
{
    // (MulDataType, SumDataType, M, N, K)
    std::vector<std::tuple<DataType, DataType, size_t, size_t, size_t>> out;

    if (!supportsExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
        return out;

    auto context_instance = avk2::InstanceContext::getGlobalInstanceContext();

    std::vector<vk::raii::PhysicalDevice> all_vk_devices = context_instance->instance().enumeratePhysicalDevices();
    rassert(device_id_vulkan < all_vk_devices.size(), 6537653743, device_id_vulkan, all_vk_devices.size());
    vk::raii::PhysicalDevice vk_device = all_vk_devices[device_id_vulkan];

    // Check feature bit to ensure the extension is actually usable
    // (some drivers may expose the extension but not the feature).
    auto feats2 = vk_device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();
    const auto coopFeat = feats2.get<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();
    if (!coopFeat.cooperativeMatrix)
        return out;

    // Enumerate all supported (A,B,C,Result) + sizes (M,N,K) for subgroup scope.
    // We expose entries keyed by the RESULT type because that is what user code
    // will typically store back to memory (e.g., FP16xFP16 -> FP32 accumulator/result).
    std::vector<vk::CooperativeMatrixPropertiesKHR> props;
    props = vk_device.getCooperativeMatrixPropertiesKHR();

    // Map result component type -> our DataType.
    auto mapToDataType = [](vk::ComponentTypeKHR t) -> std::optional<DataType> {
        switch (t) {
            case vk::ComponentTypeKHR::eFloat16: return DataType16f;
            case vk::ComponentTypeKHR::eFloat32: return DataType32f;
            case vk::ComponentTypeKHR::eSint32:  return DataType32i;
            case vk::ComponentTypeKHR::eUint32:  return DataType32u;
            case vk::ComponentTypeKHR::eUint16:  return DataType16u;
            case vk::ComponentTypeKHR::eUint8:   return DataType8u;
            // NOTE: We intentionally ignore other result types that are not modeled in our DataType set.
            default: return std::nullopt;
        }
    };

    // Use a set to deduplicate possible duplicates (some drivers may list repeats).
    std::set<std::tuple<DataType, DataType, size_t, size_t, size_t>> uniq;

    for (const auto& p : props) {
        // Only expose subgroup-scope modes; those are the baseline for KHR.
        if (p.scope != vk::ScopeKHR::eSubgroup)
            continue;

        auto aType = mapToDataType(p.AType);
        auto bType = mapToDataType(p.BType);
        auto cType = mapToDataType(p.CType);
        auto rType = mapToDataType(p.ResultType);

        if (!aType || !bType || !cType || !rType)
            continue;

        DataType mulType = *aType;
        // A and B should be of the same type (multiply type)
        if (*aType != *bType)
            continue;

        DataType sumType = *rType;
        // C and Result should be of the same type (sum/addition type)
        if (*cType != *rType)
            continue;

        uniq.emplace(mulType, sumType, static_cast<size_t>(p.MSize), static_cast<size_t>(p.NSize), static_cast<size_t>(p.KSize));
    }

    out.assign(uniq.begin(), uniq.end());
    return out;
}

bool avk2::Device::isCooperativeMatrixSizeSupported(DataType mulType, DataType accType, size_t m, size_t n, size_t k) const
{
    // typles of (MulDataType, AccDataType, M, N, K)
    const auto sizes = supportedCooperativeMatrixSizes();

    for (const auto& t : sizes) {
        if (std::get<0>(t) == mulType &&
            std::get<1>(t) == accType &&
            std::get<2>(t) == m &&
            std::get<3>(t) == n &&
            std::get<4>(t) == k)
        {
            return true;
        }
    }
    return false;
}

std::vector<DataType> avk2::Device::supportedImageDataTypes() const
{
	std::vector<DataType> formats;
	formats.push_back(DataType8u);
	formats.push_back(DataType16u);
	formats.push_back(DataType32u);
	formats.push_back(DataType32i);
	formats.push_back(DataType32f);
	return formats;
}

vk::Format avk2::Device::typeToVkFormat(DataType type) const
{
	std::vector<DataType> types = supportedImageDataTypes();
	rassert(std::find(types.begin(), types.end(), type) != types.end(), 5381124512, "Unsupported image type", typeName(type));
	if (type == DataType8u) {
		return vk::Format::eR8Unorm;
	} else if (type == DataType16u) {
		return vk::Format::eR16Unorm;
	} else if (type == DataType32u) {
		return vk::Format::eR32Uint;
	} else if (type == DataType32i) {
		return vk::Format::eR32Sint;
	} else if (type == DataType32f) {
		return vk::Format::eR32Sfloat;
	} else {
		rassert(false, 414078647, "Unsupported image type", typeName(type));
	}
}

vk::Format avk2::Device::typeToVkFormat(DataType type, size_t nchannels) const
{
	rassert(nchannels >= 1 && nchannels <= 4, 782282537, type, nchannels);
	if (type == DataType32f) {
		vk::Format formats[] = {vk::Format::eR32Sfloat,	vk::Format::eR32G32Sfloat,	vk::Format::eR32G32B32Sfloat,	vk::Format::eR32G32B32A32Sfloat};
		return formats[nchannels - 1];
	} else if (type == DataType32u) {
		vk::Format formats[] = {vk::Format::eR32Uint,	vk::Format::eR32G32Uint,	vk::Format::eR32G32B32Uint,		vk::Format::eR32G32B32A32Uint};
		return formats[nchannels - 1];
	} else {
		rassert(false, 313609583, type, nchannels);
	}
}

vk::Format avk2::Device::typeToVkDepthStencilFormat(DataType type) const
{
	std::vector<DataType> types = supportedImageDataTypes();
	rassert(std::find(types.begin(), types.end(), type) != types.end(), 451234123412, "Unsupported depth stencil image type", typeName(type));
	// DepthStencil images should have another type (eD32Sfloat insted of eR32Sfloat):
	// see adoption percent - https://vulkan.gpuinfo.org/listoptimaltilingformats.php and example of supported usages - https://vulkan.gpuinfo.org/displayreport.php?id=30317#formats
	// this is because we can't use R32Sfloat for depth stencil (have low adoption)
	// and we can't use D32Sfloat as eStorageImage and eColorAttachment (also due to low adoption)
	rassert(type == DataType32f, 736774389);
	return vk::Format::eD32Sfloat;
}

size_t avk2::Device::freeMemory()
{
	auto context_instance = avk2::InstanceContext::getGlobalInstanceContext();
	return freeMemory(context_instance->instance());
}

size_t avk2::Device::freeMemory(const vk::raii::Instance &instance)
{
	rassert(supportsFreeMemoryRequest(), 990404516703683);

	std::vector<vk::raii::PhysicalDevice> all_vk_devices = instance.enumeratePhysicalDevices();
	rassert(device_id_vulkan < all_vk_devices.size(), 2031530612780135, device_id_vulkan, all_vk_devices.size());
	vk::raii::PhysicalDevice vk_device = all_vk_devices[device_id_vulkan];

	vk::PhysicalDeviceMemoryProperties memory_properties = vk_device.getMemoryProperties();
	std::optional<unsigned int> device_local_heap_index = avk2::getIndexOfDeviceLocalHeap(memory_properties);
	rassert(device_local_heap_index.has_value(), 167503366060527);

	vk::PhysicalDeviceMemoryBudgetPropertiesEXT memory_budget_properties = vk_device.getMemoryProperties2<vk::PhysicalDeviceMemoryProperties2, vk::PhysicalDeviceMemoryBudgetPropertiesEXT>().get<vk::PhysicalDeviceMemoryBudgetPropertiesEXT>();
	size_t free_memory = memory_budget_properties.heapBudget[*device_local_heap_index];
	return free_memory;
}

void avk2::Device::printInfo()
{
	std::cout << "Using device: " << name;
	if (supportsFreeMemoryRequest()) {
		size_t free_mem_size = freeMemory();
		std::cout << ", free memory: " << (free_mem_size >> 20) << "/" << (mem_size >> 20) << " MB";
	} else {
		std::cout << ", " << (mem_size >> 20) << " MB global memory";
	}
	std::cout << ", Vulkan API " << api_version << std::endl;

	std::cout << "  driver version: " << driver_version << std::endl;
	std::cout << "  max work group size " << max_workgroup_size << std::endl;
}
