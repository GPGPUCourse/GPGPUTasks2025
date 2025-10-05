#include "engine.h"

#include "libbase/timer.h"
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libbase/runtime_assert.h>

#include "spirv_reflect/shader_module_info.h"
#include "exceptions.h"
#include "utils.h"
#include "data_buffer.h"
#include "data_image.h"

#include <atomic>
#include <filesystem>

#include "vk/common_host.h"

#include "vulkan_api_headers.h"
#if ENABLE_AND_ENSURE_RENDERDOC
	#include "renderdoc/renderdoc_app.h"
#endif

namespace {
	// see https://vulkan-tutorial.com/Drawing_a_triangle/Setup/Validation_layers
	// this is a debug callback for Vulkan Validation Layers
	// when they find any problems - this callback will be triggered
	static VKAPI_ATTR VkBool32 VKAPI_CALL
	debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
				  const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
	{
		avk2::InstanceContext *instance_context = (avk2::InstanceContext*) pUserData;

		bool message_is_suppressed = false;
		if (avk2::isMoltenVK()) {
			std::string message_id_name = std::string(pCallbackData->pMessageIdName);
			std::string message = std::string(pCallbackData->pMessage);

			std::string message_id_name_about_geometry_shader_feature = "VUID-VkShaderModuleCreateInfo-pCode-08740";
			std::string message_about_geometry_shader_feature = "SPIR-V Capability Geometry was declared, but one of the following requirements is required (VkPhysicalDeviceFeatures::geometryShader)";

			if (message_id_name == message_id_name_about_geometry_shader_feature && message.find(message_about_geometry_shader_feature) != std::string::npos) {
				message_is_suppressed = true;
				// note that for now we need geometry shader ONLY for gl_PrimitiveID
				// but there is no geometry shader support on MoltenVK,
				// so we rely on fact that it seems that gl_PrimitiveID can be used on MoltenVK even without geometry shader feature requested
				// https://computergraphics.stackexchange.com/questions/9449/vulkan-using-gl-primitiveid-without-geometryshader-feature#comment14810_9449
			}
		}

		if (!message_is_suppressed) {
			std::cerr << "Vulkan debug callback triggered with " << pCallbackData->pMessage << std::endl;
			instance_context->setDebugCallbackTriggered(true);
		} else {
			std::cout << "Vulkan debug callback triggered with suppressed message " << pCallbackData->pMessage << std::endl;
		}

		return VK_FALSE; // put debug point here to catch any validation error when it happens
	}

	VkDebugUtilsMessengerEXT setupDebugCallback(avk2::InstanceContext *instance_context)
	{
		VkDebugUtilsMessengerEXT debug_messenger;

		vk::DebugUtilsMessengerCreateInfoEXT create_info;
		vk::DebugUtilsMessageSeverityFlagsEXT any_severity = /*vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | */vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
		create_info.messageSeverity = any_severity;
		vk::DebugUtilsMessageTypeFlagsEXT any_type = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding;
		create_info.messageType = any_type;
		create_info.pfnUserCallback = debugCallback;
		create_info.pUserData = (void*) instance_context;

		rassert(VKF.vkCreateDebugUtilsMessengerEXT, 378392459011272);
		VkDebugUtilsMessengerCreateInfoEXT vk_create_info = create_info;
		VK_CHECK_RESULT(VKF.vkCreateDebugUtilsMessengerEXT(*instance_context->instance(), &vk_create_info, nullptr, &debug_messenger), 56756784764);

		return debug_messenger;
	}

	void cleanupDebugCallback(const vk::raii::Instance &instance, VkDebugUtilsMessengerEXT debug_messenger)
	{
		rassert(VKF.vkDestroyDebugUtilsMessengerEXT, 378392459011672);
		VKF.vkDestroyDebugUtilsMessengerEXT(*instance, debug_messenger, nullptr);
	}
}

bool avk2::isMoltenVK()
{
#if defined(__APPLE__)
	return true;
#else
	return false;
#endif
}

vk::ApplicationInfo avk2::createAppInfo()
{
	vk::ApplicationInfo app_info;
	app_info.setApiVersion(VULKAN_MIN_VERSION); // it should be equal to VmaAllocatorCreateInfo.vulkanApiVersion
	return app_info;
}

vk::raii::Instance avk2::createInstance(const vk::raii::Context &context, bool enable_validation_layers)
{
	std::set<std::string> context_supported_extensions;
	{
		std::vector<vk::ExtensionProperties> extension_properties = context.enumerateInstanceExtensionProperties();
		for (size_t k = 0; k < extension_properties.size(); ++k) {
			context_supported_extensions.insert(extension_properties[k].extensionName);
		}
	}

	const std::vector<const char*> validation_layers = {
			VK_LAYER_KHRONOS_VALIDATION_NAME, // this enables validation layers - they include a lot of sanity/usage correctness checks
	};
	const std::vector<vk::ValidationFeatureEnableEXT> validation_features_enabled = {
			vk::ValidationFeatureEnableEXT::eDebugPrintf, // VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
	};
	vk::ValidationFeaturesEXT validation_features(validation_features_enabled);

	vk::ApplicationInfo app_info = avk2::createAppInfo();

	vk::InstanceCreateInfo instance_create_info = {};
	instance_create_info.setPApplicationInfo(&app_info);

	std::vector<const char*> requested_extensions;

	if (enable_validation_layers) {
		rassert(!ENABLE_AND_ENSURE_RENDERDOC, 774349424); // RenderDoc can't capture frames while validation layers are in use

		std::string default_layer_dir;
#if defined(__linux__)
		default_layer_dir = "/usr/local/share/vulkan/explicit_layer.d/";
#elif defined(WIN32)
		auto supported_validation_layers_versions = {"1.3.280.0", "1.3.283.0"};
		for (std::string vulkan_sdk_version: supported_validation_layers_versions) {
			default_layer_dir = "C:\\VulkanSDK\\" + vulkan_sdk_version + "\\Bin\\";
			if (std::filesystem::exists(default_layer_dir))
				break;
		}
#elif defined(__APPLE__)
		default_layer_dir = "/opt/vulkansdk/macOS/share/vulkan/explicit_layer.d/";
#endif

		const char* layer_path = std::getenv("VK_LAYER_PATH");
		if (layer_path == nullptr && std::filesystem::exists(default_layer_dir + "VkLayer_khronos_validation.json")) {
			layer_path = default_layer_dir.c_str();
			setenv("VK_LAYER_PATH", layer_path, false);
		}
		rassert(layer_path != nullptr && !std::string(layer_path).empty(), "331505428462788 - "
				"Install Validation Layers and set VK_LAYER_PATH=" + default_layer_dir + " environment variable "
				"to specify dir with VkLayer_khronos_validation.json file inside");
		rassert(std::filesystem::exists(std::string(layer_path) + "/VkLayer_khronos_validation.json"), "331505428462789 - "
				"environment variable VK_LAYER_PATH=" + std::string(layer_path) + ", "
				"but no VkLayer_khronos_validation.json file was found in it");

		bool validation_layers_supported = false;
		for (auto layer : context.enumerateInstanceLayerProperties()) {
			if (std::string(VK_LAYER_KHRONOS_VALIDATION_NAME) == layer.layerName) {
				validation_layers_supported = true;
			}
		}
		rassert(validation_layers_supported, "12693131637537 - validation layers requested but not available");

		requested_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		instance_create_info.setEnabledLayerCount(validation_layers.size());
		instance_create_info.setPpEnabledLayerNames(validation_layers.data());

		// see https://stackoverflow.com/questions/64617959/vulkan-debugprintfext-doesnt-print-anything
		if (DEBUG_PRINTF_EXT_ENABLED) {
			instance_create_info.setPNext(&validation_features);
		}

		std::cout << "Validation Layers enabled" << std::endl;
	}

	if (isMoltenVK()) {
		if (context_supported_extensions.count(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
			// This extension allows applications to control whether devices that expose the VK_KHR_portability_subset
			// extension are included in the results of physical device enumeration.
			requested_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
			instance_create_info.setFlags(vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
		}
		if (VULKAN_MIN_VERSION_MAJOR == 1 && VULKAN_MIN_VERSION_MINOR < 1) { // if required Vulkan<1.1
			// on MacOS+MoltenVK we use VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
			// and it requires Vulkan>=1.1 or VK_KHR_get_physical_device_properties2 - see https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_portability_subset.html#_extension_and_version_dependencies
			rassert(context_supported_extensions.count(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME), 661539197);
			requested_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
		}
	}

	for (auto requested_extension: requested_extensions) {
		rassert(context_supported_extensions.count(requested_extension), "45123412412312 - instance extension " + std::string(requested_extension) + " requested but is not available");
	}

	instance_create_info.setEnabledExtensionCount(requested_extensions.size());
	instance_create_info.setPpEnabledExtensionNames(requested_extensions.data());

	if (DEBUG_PRINTF_EXT_ENABLED) {
		setenv("VK_LAYER_PRINTF_TO_STDOUT", "1", false); // DEBUG_PRINTF_TO_STDOUT is deprecated
	}

	return vk::raii::Instance(context, instance_create_info);
}

avk2::InstanceContext::InstanceContext(bool enable_validation_layers)
{
	static std::atomic<size_t> next_construction_index = 0;
	construction_index_ = next_construction_index++;

	context_ = std::make_shared<vk::raii::Context>();

	instance_ = std::make_shared<vk::raii::Instance>(avk2::createInstance(*context_, enable_validation_layers));

	// note that it is important to re-initialize dispatcher (after VKF.init(instance) in VulkanEnum::enumDevices())
	// because now was (optionally) created with enabled validation layers and debug extensions
	// f.e. this init is required for vkCreateDebugUtilsMessengerEXT loading - see rassert 378392459011272
	// also this init is required for vkCmdPushConstants loading - see rassert 315723128637936
	VKF.init(vk::Instance(*instance_));

	is_debug_callback_triggered_ = false;
	if (enable_validation_layers) {
		debug_messenger_ = std::make_shared<VkDebugUtilsMessengerEXT>(setupDebugCallback(this));
	}
}

avk2::InstanceContext::~InstanceContext()
{
	if (debug_messenger_) {
		cleanupDebugCallback(*instance_, *debug_messenger_);
		debug_messenger_.reset();
	}
	instance_.reset();
	context_.reset();
}

std::unique_ptr<std::lock_guard<std::mutex>> avk2::InstanceContext::getGlobalLock()
{
	static std::mutex global_mutex;
	return std::make_unique<std::lock_guard<std::mutex>>(global_mutex);
}

std::vector<std::shared_ptr<avk2::InstanceContext>>	avk2::InstanceContext::global_instance_contexts_(2);
bool												avk2::InstanceContext::global_rdoc_checked = false;
RENDERDOC_API_1_6_0*								avk2::InstanceContext::global_rdoc_api = nullptr;

std::shared_ptr<avk2::InstanceContext> avk2::InstanceContext::getGlobalInstanceContext(bool enable_validation_layers)
{
	auto global_instance_lock = getGlobalLock();

#if ENABLE_AND_ENSURE_RENDERDOC
	if (enable_validation_layers && ENABLE_AND_ENSURE_RENDERDOC) { // RenderDoc can't capture frames while validation layers are in use
		std::cerr << "Validation Layers were not enabled, because RenderDoc was also requested and they are incompatible" << std::endl;
		enable_validation_layers = false;
	}

	if (!global_rdoc_checked) {
		pRENDERDOC_GetAPI RENDERDOC_GetAPI = nullptr;
#ifdef WIN32
		HMODULE mod = GetModuleHandleA("renderdoc.dll");
		if (mod) {
			RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
		}
#else
	#ifdef __APPLE__
		void *mod = dlopen("librenderdoc.dylib", RTLD_NOW | RTLD_NOLOAD);
	#else
		void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD); // we don't want to search for librenderdoc.so - we want to check "is it already loaded?" (from RenderDoc GUI)
	#endif
		if (mod) {
			RENDERDOC_GetAPI = (pRENDERDOC_GetAPI) dlsym(mod, "RENDERDOC_GetAPI");
		}
#endif
		if (RENDERDOC_GetAPI) {
			int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_6_0, (void **) &global_rdoc_api);
			rassert(ret == 1, 656254195);
			rassert(global_rdoc_api, 181087328);
			std::cout << "RenderDoc API found" << std::endl;
		} else {
			std::cerr << "RenderDoc API not found - please launch application from RenderDoc GUI" << std::endl; 
		}
		global_rdoc_checked = true;
	}
#endif

	size_t index = enable_validation_layers;
	rassert(global_instance_contexts_.size() == 2, 371855177);
	if (!global_instance_contexts_[index]) {
		global_instance_contexts_[index] = std::make_shared<avk2::InstanceContext>(enable_validation_layers);
	}
	return global_instance_contexts_[index];
}

// we need to gracefully clear Vulkan context before it is too late (otherwise we encounter segfault on some systems)
void avk2::InstanceContext::clearGlobalInstanceContext()
{
	auto global_instance_lock = getGlobalLock();

	rassert(global_instance_contexts_.size() == 2, 3718551757);
	for (size_t index = 0; index < global_instance_contexts_.size(); ++index) {
		global_instance_contexts_[index].reset();
	}
}

void avk2::InstanceContext::renderDocStartCapture(const std::string &title)
{
#if ENABLE_AND_ENSURE_RENDERDOC
	if (global_rdoc_api) {
		global_rdoc_api->StartFrameCapture(nullptr, nullptr);
		global_rdoc_api->SetCaptureTitle(title.c_str());
	}
#endif
}

void avk2::InstanceContext::renderDocEndCapture()
{
#if ENABLE_AND_ENSURE_RENDERDOC
	if (global_rdoc_api) {
		rassert(global_rdoc_api, 3451412312);
		global_rdoc_api->EndFrameCapture(nullptr, nullptr);
	}
#endif
}

class avk2::VkContext {
public:
	VkContext(uint64_t vk_device_id, bool enable_validation_layers)
		: device_info_(vk_device_id), queue_family_index_(std::numeric_limits<unsigned int>::max())
	{
		instance_context_ = avk2::InstanceContext::getGlobalInstanceContext(enable_validation_layers);

		device_info_.init(instance_context_->instance());

		std::vector<vk::raii::PhysicalDevice> all_vk_devices = instance_context_->instance().enumeratePhysicalDevices();
		rassert(vk_device_id < all_vk_devices.size(), 305965578003001);
		physical_device_ = std::make_shared<vk::raii::PhysicalDevice>(all_vk_devices[vk_device_id]);

		std::optional<unsigned int> queue_family_index = avk2::getIndexOfQueueFamily(physical_device_->getQueueFamilyProperties());
		rassert(queue_family_index.has_value(), 39288697807619);
		queue_family_index_ = *queue_family_index;

		float queue_priority = 0.0f;
		vk::DeviceQueueCreateInfo queue_create_info({}, *queue_family_index, 1, &queue_priority);

		std::vector<const char*> device_enabled_extensions;
		if (DEBUG_PRINTF_EXT_ENABLED) {
			rassert(device_info_.supportsExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME), 756565853);
			device_enabled_extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
		}
		if (isMoltenVK() && device_info_.supportsExtension(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
			// Validation Error: [ VUID-VkDeviceCreateInfo-pProperties-04451 ] ... VK_KHR_portability_subset must be enabled because physical device VkPhysicalDevice 0x12f934d00[] supports it.
			// see also https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_portability_subset.html#_description
			// ... "If this extension is supported by the Vulkan implementation, the application must enable this extension." ...
			// so this is required at least for MacOS+MoltenVK
			device_enabled_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
		}

		// How to enable features - see https://github.com/KhronosGroup/Vulkan-Guide/blob/main/chapters/enabling_features.adoc#how-to-enable-the-features

		// We require this feature to use std430 layout instead of std140
		// Example of std140 drawback:
		// - float array (used in FourierCorrectionVk) is x4 times bigger due to 16 bytes strides - https://www.reddit.com/r/vulkan/comments/u5jiws/glsl_std140_layout_for_arrays_of_scalars/
		vk::PhysicalDeviceUniformBufferStandardLayoutFeatures buffer_std_layout_features(true);

		// We require VK_EXT_shader_atomic_float extension to use atomicAdd(float[], float),
		// it has wide support - https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_EXT_shader_atomic_float
		device_enabled_extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
		vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features;
		atomic_float_features.setShaderBufferFloat32AtomicAdd(true);
		atomic_float_features.setPNext(&buffer_std_layout_features);

		vk::PhysicalDeviceFeatures2 device_enabled_features2;
		if (isMoltenVK()) {
			// note that for now we need geometry shader ONLY for gl_PrimitiveID
			// but there is no geometry shader support on MoltenVK,
			// so we rely on fact that it seems that gl_PrimitiveID can be used on MoltenVK even without geometry shader feature requested
			// https://computergraphics.stackexchange.com/questions/9449/vulkan-using-gl-primitiveid-without-geometryshader-feature#comment14810_9449
		} else {
			device_enabled_features2.features.setGeometryShader(true); // requested for usage of gl_PrimitiveID
		}
		device_enabled_features2.features.setFillModeNonSolid(true); // requested for wireframe rendering (i.e. polygonMode = vk::PolygonMode::eLine)
		device_enabled_features2.setPNext(&atomic_float_features);

		vk::DeviceCreateInfo device_create_info({}, queue_create_info, {}, device_enabled_extensions, nullptr, &device_enabled_features2);
		device_ = std::make_shared<vk::raii::Device>(physical_device_->createDevice(device_create_info));

		VmaVulkanFunctions vulkanFunctions = {};
		rassert(VKF.vkGetInstanceProcAddr && physical_device_->getDispatcher()->vkGetDeviceProcAddr, 755371085912201);
		vulkanFunctions.vkGetInstanceProcAddr = VKF.vkGetInstanceProcAddr;
		vulkanFunctions.vkGetDeviceProcAddr = physical_device_->getDispatcher()->vkGetDeviceProcAddr;
		VmaAllocatorCreateInfo allocatorCreateInfo = {};
		allocatorCreateInfo.flags = 0;
		if (device_info_.supportsFreeMemoryRequest()) {
			allocatorCreateInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
		}
		// see https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/quick_start.html
		allocatorCreateInfo.vulkanApiVersion = VULKAN_MIN_VERSION; // it should be equal to vk::ApplicationInfo.apiVersion
		allocatorCreateInfo.physicalDevice = vk::PhysicalDevice(*physical_device_.get());
		allocatorCreateInfo.device = vk::Device(*device_.get());
		allocatorCreateInfo.instance = vk::Instance(*instance_context_->instance());
		allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
		vma_ = std::shared_ptr<VmaAllocator>(new VmaAllocator());
		VK_CHECK_RESULT(vmaCreateAllocator(&allocatorCreateInfo, vma_.get()), 56748938637);

		queue_ = std::make_shared<vk::raii::Queue>(device_->getQueue(*queue_family_index, 0));

		vk::CommandPoolCreateInfo command_pool_create_info({}, *queue_family_index);
		command_pool_ = std::make_shared<vk::raii::CommandPool>(device_->createCommandPool(command_pool_create_info));

		std::vector<vk::DescriptorPoolSize> pool_size_per_type;
		for (vk::DescriptorType type: VK_POOL_DESCRIPTOR_TYPES) {
			pool_size_per_type.push_back(vk::DescriptorPoolSize(type, VK_MAX_DESCRIPTORS_PER_TYPE));
		}
		unsigned int max_descriptor_sets = 2; // because we have special descriptor set for rassert - see VK_RASSERT_CODE_SET usage
		vk::DescriptorPoolCreateInfo descriptor_pool_create_info(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, max_descriptor_sets, pool_size_per_type);
		descriptor_pool_ = std::make_shared<vk::raii::DescriptorPool>(device_->createDescriptorPool(descriptor_pool_create_info));
	}

	~VkContext()
	{
		descriptor_pool_.reset();
		command_pool_.reset();
		queue_.reset();
		if (vma_) {
			vmaDestroyAllocator(*vma_);
			vma_.reset();
		}
		device_.reset();
		queue_family_index_ = std::numeric_limits<unsigned int>::max();
		physical_device_.reset();
		instance_context_.reset();
	}

	avk2::Device								device_info_;

	std::shared_ptr<avk2::InstanceContext>		instance_context_;
	std::shared_ptr<vk::raii::PhysicalDevice>	physical_device_;
	unsigned int								queue_family_index_;
	std::shared_ptr<vk::raii::Device>			device_;
	std::shared_ptr<VmaAllocator>				vma_;
	std::shared_ptr<vk::raii::Queue>			queue_;
	std::shared_ptr<vk::raii::CommandPool>		command_pool_;
	std::shared_ptr<vk::raii::DescriptorPool>	descriptor_pool_; // TODO allocate them when needed with exponentially increasing size? (on per-type basis) or even use VK_KHR_push_descriptors
};

vk::raii::Context &avk2::VulkanEngine::getContext()
{
	rassert(avk2_context_ && avk2_context_->instance_context_, 352524414255702);
	return avk2_context_->instance_context_->context();
}

vk::raii::Instance &avk2::VulkanEngine::getInstance()
{
	rassert(avk2_context_ && avk2_context_->instance_context_, 352524414257502);
	return avk2_context_->instance_context_->instance();
}

vk::raii::PhysicalDevice &avk2::VulkanEngine::getPhysicalDevice()
{
	rassert(avk2_context_ && avk2_context_->physical_device_, 3464324414255702);
	return *avk2_context_->physical_device_;
}

vk::raii::Device &avk2::VulkanEngine::getDevice()
{
	rassert(avk2_context_ && avk2_context_->device_, 823104721289038);
	return *avk2_context_->device_;
}

VmaAllocator &avk2::VulkanEngine::getVma()
{
	rassert(avk2_context_ && avk2_context_->vma_, 823104721289036);
	return *avk2_context_->vma_;
}

const unsigned int &avk2::VulkanEngine::getQueueFamilyIndex()
{
	return avk2_context_->queue_family_index_;
}

vk::raii::Queue &avk2::VulkanEngine::getQueue()
{
	rassert(avk2_context_ && avk2_context_->queue_, 769450975410122);
	return *avk2_context_->queue_;
}

vk::raii::CommandPool &avk2::VulkanEngine::getCommandPool()
{
	rassert(avk2_context_ && avk2_context_->command_pool_, 935441028376960);
	return *avk2_context_->command_pool_;
}

vk::raii::CommandBuffer avk2::VulkanEngine::createCommandBuffer()
{
	vk::CommandBufferAllocateInfo command_buffer_allocate_info(getCommandPool(), vk::CommandBufferLevel::ePrimary, 1);
	vk::raii::CommandBuffer command_buffer = std::move(vk::raii::CommandBuffers(getDevice(), command_buffer_allocate_info)[0]);
	return command_buffer;
}

void avk2::VulkanEngine::submitCommandBuffer(const vk::raii::CommandBuffer &command_buffer)
{
	std::shared_ptr<vk::raii::Fence> fence = submitCommandBufferAsync(command_buffer);
	VK_CHECK_RESULT(getDevice().waitForFences(vk::Fence(*fence), true, VULKAN_TIMEOUT_NANOSECS), 345123451241);
}

std::shared_ptr<vk::raii::Fence> avk2::VulkanEngine::submitCommandBufferAsync(const vk::raii::CommandBuffer &command_buffer)
{
	vk::CommandBuffer command_buffer_non_raii = command_buffer;
	std::shared_ptr<vk::raii::Fence> fence = std::make_shared<vk::raii::Fence>(getDevice(), vk::FenceCreateInfo());

	vk::SubmitInfo submit_info(nullptr, nullptr, command_buffer_non_raii);
	getQueue().submit({submit_info}, *fence);

	return fence;
}

vk::raii::DescriptorSet avk2::VulkanEngine::allocateDescriptor(vk::raii::DescriptorSetLayout& descriptor_set_layout, const std::vector<vk::DescriptorType> &descriptor_types)
{
	std::vector<vk::DescriptorType> supported_types = VK_POOL_DESCRIPTOR_TYPES;
	for (auto type: descriptor_types) {
		rassert(std::find(supported_types.begin(), supported_types.end(), type) != supported_types.end(), 389005302, to_string(type));
	}
	vk::DescriptorSetLayout descriptor_set_layout_non_raii = descriptor_set_layout;
	vk::DescriptorSetAllocateInfo allocate_info(getDescriptorPool(), descriptor_set_layout_non_raii);
	std::vector<vk::raii::DescriptorSet> descriptor_sets = avk2_context_->device_->allocateDescriptorSets(allocate_info);
	rassert(descriptor_sets.size() == 1, 764756099177612);
	return std::move(descriptor_sets[0]);
}

vk::raii::DescriptorPool &avk2::VulkanEngine::getDescriptorPool()
{
	rassert(avk2_context_ && avk2_context_->descriptor_pool_, 26077712935129);
	return *avk2_context_->descriptor_pool_;
}

avk2::VulkanEngine::VulkanEngine()
	: vk_device_id_(std::numeric_limits<uint64_t>::max()), device_(vk_device_id_)
{}

avk2::VulkanEngine::~VulkanEngine()
{
	if (kernels_.size() != 0) {
		std::cerr << "VulkanEngine: uncleared kernel found" << std::endl;
	}
}

void avk2::VulkanEngine::init(uint64_t vk_device_id, bool enable_validation_layers)
{
	vk_device_id_ = vk_device_id;

	avk2_context_ = std::make_shared<avk2::VkContext>(vk_device_id, enable_validation_layers);

#if DEBUG_PRINTF_EXT_ENABLED
	if (enable_validation_layers) {
		std::cout << "Debug printf is enabled" << std::endl;
	} else {
		std::cerr << "Debug printf was requested, but validation layers were disabled so debug printf will not work, "
					 "please enable validation layers with context.setVKValidationLayers(true) after calling context.initVulkan(...)" << std::endl;
	}
#endif

	device_ = Device(vk_device_id_);
	bool inited_ok = device_.init(*avk2_context_->instance_context_->instance());
	rassert(inited_ok, 629844440093078);

	allocateStagingWriteBuffers();
	allocateStagingReadBuffers();
}

avk2::raii::BufferData* avk2::VulkanEngine::createBuffer(size_t size)
{
	rassert(size > 0, 435172894312);
	vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer;
	usage_flags |= vk::BufferUsageFlagBits::eTransferDst; // required for usage as DST in copyBuffer(src, DST) - i.e. for shared_device_buffer::write
	usage_flags |= vk::BufferUsageFlagBits::eTransferSrc; // required for usage as SRC in copyBuffer(SRC, dst) - i.e. for shared_device_buffer::read
	usage_flags |= vk::BufferUsageFlagBits::eVertexBuffer;// required for usage as vertices buffer - i.e. can be used via vkCmdBindVertexBuffers
	usage_flags |= vk::BufferUsageFlagBits::eIndexBuffer; // required for usage as indices (i.e. faces) buffer - i.e. can be used via vkCmdBindIndexBuffer
	vk::BufferCreateInfo raii_buffer_create_info{vk::BufferCreateFlags(), size, usage_flags, vk::SharingMode::eExclusive, 1, &getQueueFamilyIndex()};
	VkBufferCreateInfo buffer_create_info = raii_buffer_create_info;

	VmaAllocationCreateInfo allocation_info = {};
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	allocation_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

	VkBuffer buffer;
	VmaAllocation buffer_allocation;
	VK_CHECK_RESULT(vmaCreateBuffer(getVma(), &buffer_create_info, &allocation_info, &buffer, &buffer_allocation, nullptr), 308379010504948);

	return new avk2::raii::BufferData(buffer, buffer_allocation);
}

avk2::raii::ImageData* avk2::VulkanEngine::createDepthImage(unsigned int width, unsigned int height)
{
	const vk::Format format = device_.typeToVkDepthStencilFormat(DataType32f);
	return createImage2D(width, height, format);
}

avk2::raii::ImageData* avk2::VulkanEngine::createImage2DArray(unsigned int width, unsigned int height, size_t cn, DataType data_type)
{
	const vk::Format format = device_.typeToVkFormat(data_type);
	return createImage2DArray(width, height, cn, format);
}

avk2::raii::ImageData* avk2::VulkanEngine::createImage2D(unsigned int width, unsigned int height, vk::Format format)
{
	return createImage2DArray(width, height, 1, format); // TODO use Image2D instead of Image2DArray
}

avk2::raii::ImageData* avk2::VulkanEngine::createImage2DArray(unsigned int width, unsigned int height, size_t cn, vk::Format format)
{
	rassert(width > 0 && height > 0, 537424666);
	rassert(cn >= 1, 975169407);
	size_t nlayers = cn;
	vk::ImageLayout initial_layout = vk::ImageLayout::eUndefined;

	vk::Format depth_format = device_.typeToVkDepthStencilFormat(DataType32f);
	bool isDepthImage = (format == depth_format);

	// see https://github.com/SaschaWillems/Vulkan/blob/52779a1bd1e6089ca3b172612e17d252ab480da4/examples/computeshader/computeshader.cpp#L103
	bool image_format_is_supported = false;
	for (auto type: device_.supportedImageDataTypes()) {
		if (format == device_.typeToVkFormat(type)) {
			image_format_is_supported = true;
		}
	}
	rassert(image_format_is_supported || format == depth_format, 116330000);

	vk::FormatProperties format_properties = getPhysicalDevice().getFormatProperties(format);

	// creating Image https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageUsageFlagBits.html
	vk::ImageUsageFlags usage_flags = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	// see https://vulkan.gpuinfo.org/displayreport.php?id=30317#formats about compatibility between usage flags and vk::Format
	if (isDepthImage) {
		usage_flags |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
	} else {
		usage_flags |= vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment;
		//			| vk::ImageUsageFlagBits::eInputAttachment // add support if needed
		//			| ...
	}
	if (device_.max_image_dimension_2d < std::max(width, height)) {
		throw avk2::vk_exception("Vulkan device " + device_.name + " can't create image2DArray "
								 + to_string(nlayers) + "x" + to_string(width) + "x" + to_string(height) + " (format=" + to_string(format) + ")"
								 + " because maximum image dimension is " + to_string(device_.max_image_dimension_2d));
	}
	vk::ImageCreateInfo raii_image_create_info(vk::ImageCreateFlags(), vk::ImageType::e2D,
											   format, {width, height, 1}, 1, nlayers,
											   vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, usage_flags, vk::SharingMode::eExclusive,
											   1, &getQueueFamilyIndex(), initial_layout);
	// TODO ADD LAYOUTS SUPPORT: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
	VkImageCreateInfo image_create_info = raii_image_create_info;

	VmaAllocationCreateInfo allocation_info = {};
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	allocation_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

	VkImage image;
	VmaAllocation image_allocation;
	VK_CHECK_RESULT(vmaCreateImage(getVma(), &image_create_info, &allocation_info, &image, &image_allocation, nullptr), 43514123421321);

	// creating Sampler https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSamplerCreateInfo.html
	vk::Filter filter_mode = vk::Filter::eLinear;
	vk::SamplerMipmapMode mipmap_mode = vk::SamplerMipmapMode::eLinear;
	vk::SamplerAddressMode address_mode = vk::SamplerAddressMode::eClampToEdge;
	vk::Bool32 unnormalized_coordinates = vk::False;
	vk::SamplerCreateInfo sampler_create_info(vk::SamplerCreateFlags(), filter_mode, filter_mode, mipmap_mode, address_mode, address_mode, address_mode, 0.0f,
											  vk::False, 0.0f, vk::False, vk::CompareOp::eNever, 0.0f, 0.0f, vk::BorderColor::eFloatTransparentBlack, unnormalized_coordinates);
	vk::raii::Sampler sampler = getDevice().createSampler(sampler_create_info);

	vk::ImageAspectFlags image_aspect = (isDepthImage ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor);
	return new avk2::raii::ImageData(image, image_allocation, std::move(sampler), initial_layout, (VkFlags) image_aspect, format,
									 width, height, cn);
}

template <typename T>
void avk2::VulkanEngine::writeImage(const avk2::raii::ImageData &image_dst, const TypedImage<T> &src)
{
	Lock staging_buffers_lock(staging_write_buffers_mutex_);

	size_t width = src.width();
	size_t height = src.height();
	size_t cn = src.channels();
	size_t size = height * width * cn * sizeof(T);

	rassert(image_dst.width() >= src.width(), 458383245638);
	rassert(image_dst.height() >= src.height(), 4583832345628);
	rassert(image_dst.channels() == src.channels(), 324124124321);

	// we are going to transfer image chunk by chunk via intermediate pre-allocated staging buffer like in writeBuffer
	// in such way we are close enough to full PCI-E utilization
	// thanks to zero-allocation and interleaving between memcpy and PCI-E transfer
	size_t row_size = width * 1 * sizeof(T);
	size_t nrows_per_chunk = STAGING_BUFFER_SIZE / row_size;
	// if this fails - we need to increase STAGING_BUFFER_SIZE so that at least one row always could fit in it
	rassert(row_size <= STAGING_BUFFER_SIZE && nrows_per_chunk >= 1, 249472063); 
	size_t nchunks = div_ceil(height, nrows_per_chunk);

	for (size_t c = 0; c < cn; ++c) {
		for (ptrdiff_t chunk_i = 0; chunk_i <= nchunks; ++chunk_i) {
			bool is_cur_chunk_exists = chunk_i < nchunks;
			bool is_prev_chunk_exists = chunk_i > 0;

			std::shared_ptr<vk::raii::CommandBuffer> command_buffer;
			std::shared_ptr<vk::raii::Fence> fence;

			if (is_prev_chunk_exists) {
				// read data from the previous staging buffer into VRAM GPU buffer
				size_t prev_buffer = (chunk_i - 1) % 2;
				size_t prev_chunk_row_start = (chunk_i - 1) *  nrows_per_chunk;
				size_t prev_chunk_row_end = std::min(prev_chunk_row_start + nrows_per_chunk, height);
	
				command_buffer = std::make_shared<vk::raii::CommandBuffer>(createCommandBuffer());
				command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
				vk::ImageSubresourceLayers image_subresource_layers(vk::ImageAspectFlags(image_dst.getAspectFlags()), 0, c, 1);
				command_buffer->copyBufferToImage(staging_write_buffers_[prev_buffer]->getBuffer(), image_dst.getImage(), image_dst.getCurrentLayout(),
												  vk::BufferImageCopy(0, width, prev_chunk_row_end - prev_chunk_row_start, image_subresource_layers,
																	  vk::Offset3D(0, prev_chunk_row_start, 0), vk::Extent3D(width, prev_chunk_row_end - prev_chunk_row_start, 1)));
				command_buffer->end();

				fence = submitCommandBufferAsync(*command_buffer);
			}

			if (is_cur_chunk_exists) {
				size_t cur_buffer = (chunk_i % 2);
				size_t cur_chunk_row_start = chunk_i * nrows_per_chunk;
				size_t cur_chunk_row_end = std::min(cur_chunk_row_start + nrows_per_chunk, height);

				// we are copying data into the current staging buffer
				// while previous staging buffer is writing into VRAM in async fashion

				T *staging_buffer_dst = (T*) staging_write_buffers_[cur_buffer]->getMappedDataPointer();
				for (ptrdiff_t j = cur_chunk_row_start; j < cur_chunk_row_end; ++j) {
					const T *from = (T*) src.ptr() + (j * width * cn + c);
					T *to = staging_buffer_dst + ((j - cur_chunk_row_start) * width);
					for (ptrdiff_t i = 0; i < width; ++i) {
						// staging_buffer_dst[(j - cur_chunk_row_start) * width + i] = ((T*) src.ptr())[j * width * cn + i * cn + c];
						*to = *from;
						from += cn;
						to += 1;
					}
				}
			}

			if (is_prev_chunk_exists) {
				VK_CHECK_RESULT(getDevice().waitForFences(vk::Fence(*fence), true, VULKAN_TIMEOUT_NANOSECS), 4512341231);
			}
		}
	}
}

void avk2::VulkanEngine::writeImage(const avk2::raii::ImageData &image_dst, const AnyImage &src)
{
	switch (src.type()) {
	case DataType8i:	writeImage<char>				(image_dst, src); break;
	case DataType8u:	writeImage<unsigned char>		(image_dst, src); break;
	case DataType16i:	writeImage<short>				(image_dst, src); break;
	case DataType16u:	writeImage<unsigned short>		(image_dst, src); break;
	case DataType32i:	writeImage<int>					(image_dst, src); break;
	case DataType32u:	writeImage<unsigned int>		(image_dst, src); break;
	case DataType32f:	writeImage<float>				(image_dst, src); break;
	case DataType64i:	writeImage<long long>			(image_dst, src); break;
	case DataType64u:	writeImage<unsigned long long>	(image_dst, src); break;
	case DataType64f:	writeImage<double>				(image_dst, src); break;
	default:			throwUnsupportedDataType(src.type());
	}
}

template <typename T>
void avk2::VulkanEngine::readImage(const avk2::raii::ImageData &image_src, const TypedImage<T> &dst)
{
	Lock staging_buffers_lock(staging_read_buffers_mutex_);

	size_t width = dst.width();
	size_t height = dst.height();
	size_t cn = dst.channels();
	size_t size = height * width * cn * sizeof(T);

	rassert(image_src.width() >= dst.width(), 458383245623138);
	rassert(image_src.height() >= dst.height(), 45838323451628);
	rassert(image_src.channels() == dst.channels(), 324124153424321);

	// we are going to transfer image chunk by chunk via intermediate pre-allocated staging buffer like in writeBuffer
	// in such way we are close enough to full PCI-E utilization
	// thanks to zero-allocation and interleaving between memcpy and PCI-E transfer
	size_t row_size = width * 1 * sizeof(T);
	size_t nrows_per_chunk = STAGING_BUFFER_SIZE / row_size;
	// if this fails - we need to increase STAGING_BUFFER_SIZE so that at least one row always could fit in it
	rassert(row_size <= STAGING_BUFFER_SIZE && nrows_per_chunk >= 1, 249472063);
	size_t nchunks = div_ceil(height, nrows_per_chunk);

	for (size_t c = 0; c < cn; ++c) {
		for (ptrdiff_t chunk_i = 0; chunk_i <= nchunks; ++chunk_i) {
			bool is_cur_chunk_exists = chunk_i < nchunks;
			bool is_prev_chunk_exists = chunk_i > 0;

			std::shared_ptr<vk::raii::CommandBuffer> command_buffer;
			std::shared_ptr<vk::raii::Fence> fence;

			if (is_cur_chunk_exists) {
				// read data from the VRAM GPU image into the staging buffer
				size_t cur_buffer = (chunk_i % 2);
				size_t cur_chunk_row_start = chunk_i * nrows_per_chunk;
				size_t cur_chunk_row_end = std::min(cur_chunk_row_start + nrows_per_chunk, height);

				command_buffer = std::make_shared<vk::raii::CommandBuffer>(createCommandBuffer());
				command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
				vk::ImageSubresourceLayers image_subresource_layers(vk::ImageAspectFlags(image_src.getAspectFlags()), 0, c, 1);
				command_buffer->copyImageToBuffer(image_src.getImage(), image_src.getCurrentLayout(), staging_read_buffers_[cur_buffer]->getBuffer(),
												  vk::BufferImageCopy(0, width, cur_chunk_row_end - cur_chunk_row_start, image_subresource_layers,
																	  vk::Offset3D(0, cur_chunk_row_start, 0), vk::Extent3D(width, cur_chunk_row_end - cur_chunk_row_start, 1)));
				command_buffer->end();

				fence = submitCommandBufferAsync(*command_buffer);
			}

			if (is_prev_chunk_exists) {
				// we are copying data from the previous staging buffer
				// while current staging buffer is filling from VRAM in async fashion
				size_t prev_buffer = ((chunk_i - 1) % 2);
				size_t prev_chunk_row_start = (chunk_i - 1) *  nrows_per_chunk;
				size_t prev_chunk_row_end = std::min(prev_chunk_row_start + nrows_per_chunk, height);

				T *staging_buffer_src = (T*) staging_read_buffers_[prev_buffer]->getMappedDataPointer();
				for (ptrdiff_t j = prev_chunk_row_start; j < prev_chunk_row_end; ++j) {
					T *to = (T*) dst.ptr() + (j * width * cn + c);
					const T *from = staging_buffer_src + ((j - prev_chunk_row_start) * width);
					for (ptrdiff_t i = 0; i < width; ++i) {
						// ((T*) dst.ptr())[j * width * cn + i * cn + c] = staging_buffer_src[(j - prev_chunk_row_start) * width + i];
						*to = *from;
						from += 1;
						to += cn;
					}
				}
			}

			if (is_cur_chunk_exists) {
				VK_CHECK_RESULT(getDevice().waitForFences(vk::Fence(*fence), true, VULKAN_TIMEOUT_NANOSECS), 34124125123);
			}
		}
	}
}

void avk2::VulkanEngine::readImage(const avk2::raii::ImageData &image_src, const AnyImage &dst)
{
	switch (dst.type()) {
	case DataType8i:	readImage<char>					(image_src, dst); break;
	case DataType8u:	readImage<unsigned char>		(image_src, dst); break;
	case DataType16i:	readImage<short>				(image_src, dst); break;
	case DataType16u:	readImage<unsigned short>		(image_src, dst); break;
	case DataType32i:	readImage<int>					(image_src, dst); break;
	case DataType32u:	readImage<unsigned int>			(image_src, dst); break;
	case DataType32f:	readImage<float>				(image_src, dst); break;
	case DataType64i:	readImage<long long>			(image_src, dst); break;
	case DataType64u:	readImage<unsigned long long>	(image_src, dst); break;
	case DataType64f:	readImage<double>				(image_src, dst); break;
	default:			throwUnsupportedDataType(dst.type());
	}
}

void avk2::VulkanEngine::allocateStagingWriteBuffers()
{
	// allocating HOST-visible staging buffer
	// https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html#usage_patterns_readback
	VkBufferCreateInfo buffer_create_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	buffer_create_info.size = STAGING_BUFFER_SIZE;
	buffer_create_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	VmaAllocationCreateInfo allocation_info = {};
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
	allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

	for (int i = 0; i < 2; ++i) {
		VkBuffer staging_buffer;
		VmaAllocation staging_buffer_allocation;
		VmaAllocationInfo staging_alloc_info;
		VK_CHECK_RESULT(vmaCreateBuffer(getVma(), &buffer_create_info, &allocation_info, &staging_buffer, &staging_buffer_allocation, &staging_alloc_info), 30837901050495318);
		staging_write_buffers_[i] = std::unique_ptr<avk2::raii::BufferData>(new avk2::raii::BufferData(staging_buffer, staging_buffer_allocation, staging_alloc_info));
	}
}

void avk2::VulkanEngine::allocateStagingReadBuffers()
{
	// allocating HOST-visible staging buffer
	// https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html#usage_patterns_readback
	VkBufferCreateInfo buffer_create_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	buffer_create_info.size = STAGING_BUFFER_SIZE;
	buffer_create_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	VmaAllocationCreateInfo allocation_info = {};
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
	allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;

	for (int i = 0; i < 2; ++i) {
		VkBuffer staging_buffer;
		VmaAllocation staging_buffer_allocation;
		VmaAllocationInfo staging_alloc_info;
		VK_CHECK_RESULT(vmaCreateBuffer(getVma(), &buffer_create_info, &allocation_info, &staging_buffer, &staging_buffer_allocation, &staging_alloc_info), 308375350495318);
		staging_read_buffers_[i] = std::unique_ptr<avk2::raii::BufferData>(new avk2::raii::BufferData(staging_buffer, staging_buffer_allocation, staging_alloc_info));
	}
}

void avk2::VulkanEngine::writeBuffer(const avk2::raii::BufferData &buffer_dst, size_t offset, size_t size, const void *src)
{
	Lock staging_buffers_lock(staging_write_buffers_mutex_);

	// we re-use two staged buffers for double buffering-like scheme (to do memcpy and PCI-E transfer in parallel in async fashion)
	size_t chunk_size = STAGING_BUFFER_SIZE;
	size_t nchunks = div_ceil(size, chunk_size);

	for (ptrdiff_t chunk_i = 0; chunk_i <= nchunks; ++chunk_i) {
		bool is_cur_chunk_exists = chunk_i < nchunks;
		bool is_prev_chunk_exists = chunk_i > 0;

		std::shared_ptr<vk::raii::CommandBuffer> command_buffer;
		std::shared_ptr<vk::raii::Fence> fence;

		if (is_prev_chunk_exists) {
			// read data from the previous staging buffer into VRAM GPU buffer
			size_t prev_buffer = (chunk_i - 1) % 2;
			size_t prev_chunk_start = (chunk_i - 1) * chunk_size;
			size_t prev_chunk_end = std::min(prev_chunk_start + chunk_size, size);

			command_buffer = std::make_shared<vk::raii::CommandBuffer>(createCommandBuffer());
			command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			command_buffer->copyBuffer(staging_write_buffers_[prev_buffer]->getBuffer(), buffer_dst.getBuffer(), {{(size_t) 0, offset + prev_chunk_start, prev_chunk_end - prev_chunk_start}});
			command_buffer->end();

			fence = submitCommandBufferAsync(*command_buffer);
		}

		if (is_cur_chunk_exists) {
			size_t cur_buffer = (chunk_i % 2);
			size_t cur_chunk_start = chunk_i * chunk_size;
			size_t cur_chunk_end = std::min(cur_chunk_start + chunk_size, size);

			// we are copying data into the current staging buffer
			// while previous staging buffer is writing into VRAM in async fashion
			memcpy((char *) staging_write_buffers_[cur_buffer]->getMappedDataPointer(), (const char*) src + cur_chunk_start, cur_chunk_end - cur_chunk_start);
		}

		if (is_prev_chunk_exists) {
			VK_CHECK_RESULT(getDevice().waitForFences(vk::Fence(*fence), true, VULKAN_TIMEOUT_NANOSECS), 453151251236);
		}
	}
}

void avk2::VulkanEngine::readBuffer(const avk2::raii::BufferData &buffer_src, size_t offset, size_t size, void *dst)
{
	Lock staging_buffers_lock(staging_read_buffers_mutex_);

	// we re-use two staged buffers for double buffering-like scheme (to do memcpy and PCI-E transfer in parallel in async fashion)
	size_t chunk_size = STAGING_BUFFER_SIZE;
	size_t nchunks = div_ceil(size, chunk_size);

	for (ptrdiff_t chunk_i = 0; chunk_i <= nchunks; ++chunk_i) {
		bool is_cur_chunk_exists = chunk_i < nchunks;
		bool is_prev_chunk_exists = chunk_i > 0;

		std::shared_ptr<vk::raii::CommandBuffer> command_buffer;
		std::shared_ptr<vk::raii::Fence> fence;

		if (is_cur_chunk_exists) {
			// read data from VRAM GPU buffer into the staging buffer
			size_t cur_buffer = (chunk_i % 2);
			size_t cur_chunk_start = chunk_i * chunk_size;
			size_t cur_chunk_end = std::min(cur_chunk_start + chunk_size, size);

			command_buffer = std::make_shared<vk::raii::CommandBuffer>(createCommandBuffer());
			command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
			command_buffer->copyBuffer(buffer_src.getBuffer(), staging_read_buffers_[cur_buffer]->getBuffer(), {{offset + cur_chunk_start, (size_t) 0, cur_chunk_end - cur_chunk_start}});
			command_buffer->end();

			fence = submitCommandBufferAsync(*command_buffer);
		}

		if (is_prev_chunk_exists) {
			// we are copying data from the previous staging buffer
			// while current staging buffer is filling from VRAM in async fashion
			size_t prev_buffer = ((chunk_i - 1) % 2);
			size_t prev_chunk_start = (chunk_i - 1) * chunk_size;
			size_t prev_chunk_end = std::min(prev_chunk_start + chunk_size, size);
			memcpy((char*) dst + prev_chunk_start, staging_read_buffers_[prev_buffer]->getMappedDataPointer(), prev_chunk_end - prev_chunk_start);
		}

		if (is_cur_chunk_exists) {
			VK_CHECK_RESULT(getDevice().waitForFences(vk::Fence(*fence), true, VULKAN_TIMEOUT_NANOSECS), 675623543242141);
		}
	}
}

avk2::VulkanKernel *avk2::VulkanEngine::findKernel(int id) const
{
	std::map<int, VulkanKernel *>::const_iterator it = kernels_.find(id);
	if (it != kernels_.end())
		return it->second;
	return nullptr;
}

void avk2::VulkanEngine::clearKernel(int id)
{
	auto it = kernels_.find(id);
	if (it != kernels_.end()) {
		delete it->second;
		kernels_.erase(it);
	}
}

void avk2::VulkanEngine::clearKernels()
{
	for (auto it = kernels_.begin(); it != kernels_.end(); ++it)
		delete it->second;
	kernels_.clear();
}

void avk2::VulkanEngine::clearStagingBuffers()
{
	for (int i = 0; i < 2; ++i) {
		staging_write_buffers_[i].reset();
		staging_read_buffers_[i].reset();
	}
}

avk2::VersionedBinary::VersionedBinary(const char *data, const size_t size)
		: data_(data), size_(size)
{}

avk2::ProgramBinaries::ProgramBinaries(std::vector<const VersionedBinary*> binaries, std::string program_name)
{
	static int next_program_id = 0;
	program_name_ = program_name;
	id_			= next_program_id++;
	binaries_	= binaries;
}

const avk2::VersionedBinary* avk2::ProgramBinaries::getBinary() const
{
	rassert(binaries_.size() > 0, 101196811550650);
	for (size_t i = 0; i < binaries_.size(); ++i) {
		const VersionedBinary* binary = binaries_[i];
		return binary;
	}

	throw vk_exception("No appropriate Vulkan program version");
}

bool avk2::ProgramBinaries::isProgramNameEndsWith(const std::string &suffix) const
{
	return ends_with(program_name_, suffix);
}

avk2::KernelSource::KernelSource(const ProgramBinaries &compute_shader_program)
: last_exec_total_time_(0.0), last_exec_prepairing_time_(0.0), last_exec_gpu_time_(0.0)
{
	shaders_programs_.push_back(&compute_shader_program);

	init();

	rassert(isCompute(), 98364576506);
}

avk2::KernelSource::KernelSource(const std::vector<const ProgramBinaries*> &graphic_shaders_programs)
: last_exec_total_time_(0.0), last_exec_prepairing_time_(0.0), last_exec_gpu_time_(0.0)
{
	shaders_programs_ = graphic_shaders_programs;

	init();
}

avk2::KernelSource::~KernelSource()
{
	gpu::Context context;
	if (context.type() == gpu::Context::TypeVulkan) {
		sh_ptr_vk_engine vk = context.vk();
		context.vk()->clearKernel(id_);
		// without this - if KernelSource is declared in local scope - then its VulkanKernel counterpart
		// will live until Vulkan context is not destroyed
		// and if the local scope is visited again - new KernelSource construction will lead to new VulkanKernel in addition to the old one
	}
}

void avk2::KernelSource::init()
{
	id_		= getNextKernelId();

	// note that glslc doesn't support multiple entry points - https://github.com/KhronosGroup/glslang/issues/605
	// note that glslc support different from "main" entry point name - https://github.com/KhronosGroup/glslang/issues/1045 
	name_	= "main";
}

int avk2::KernelSource::getNextKernelId()
{
	static int next_kernel_id = 0;
	return next_kernel_id++;
}

namespace avk2 {
	std::vector<vk::DescriptorSetLayoutBinding> createDescriptorSetLayoutBindings(const std::vector<vk::DescriptorType> &descriptors_types, vk::ShaderStageFlags stage_flags)
	{
		std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_bindings;
		for (size_t i = 0; i < descriptors_types.size(); ++i) {
			unsigned int binding = i;
			descriptor_set_layout_bindings.push_back(vk::DescriptorSetLayoutBinding(binding, descriptors_types[i], 1, stage_flags));
		}
		return descriptor_set_layout_bindings;
	}
}

vk::raii::ShaderModule avk2::KernelSource::createShaderModule(const std::shared_ptr<VulkanEngine> &vk, const ProgramBinaries &program, avk2::ShaderModuleInfo *shader_module_info_output)
{
	std::string program_name = program.programName();
	std::string program_data = std::string((const char*)program.getBinary()->data(), program.getBinary()->size());

	// https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkShaderModuleCreateInfo.html
	// > If pCode is a pointer to SPIR-V code, codeSize must be a multiple of 4
	rassert(program_data.size() % 4 == 0, 345539135345180);
	vk::ShaderModuleCreateInfo shader_module_create_info(vk::ShaderModuleCreateFlags(), program_data.size(), (const uint32_t*) program_data.data());
	vk::raii::ShaderModule shader_module = vk->getDevice().createShaderModule(shader_module_create_info);

	if (shader_module_info_output) {
		// using https://github.com/KhronosGroup/SPIRV-Reflect we can get info from kernel SPIR-V assembly
		*shader_module_info_output = avk2::ShaderModuleInfo(program_data, program_name);
	}

	return std::move(shader_module);
}

avk2::VulkanKernel *avk2::KernelSource::compileComputeKernel(const std::shared_ptr<VulkanEngine> &vk) {
	const ProgramBinaries* compute_program = shaders_programs_[0];
	rassert(compute_program->isProgramNameEndsWith("_comp"), 350141882);

	avk2::ShaderModuleInfo shader_module_info;
	vk::raii::ShaderModule shader_module = createShaderModule(vk, *compute_program, &shader_module_info);

	vk::PipelineShaderStageCreateInfo pipeline_stages_create_info({}, vk::ShaderStageFlagBits::eCompute, shader_module, name_.c_str());

	std::set<unsigned int> descriptors_sets = shader_module_info.getDescriptorsSets();
	// let's check that common.vk (and so rassert.vk) was included from Vulkan kernel:
	// #include <libgpu/vulkan/vk/common.vk>
	rassert(descriptors_sets.count(VK_MAIN_BINDING_SET) > 0, "No main descriptors set=" + to_string(VK_MAIN_BINDING_SET) + " for kernel layout found");

	std::vector<vk::DescriptorType> descriptor_types = avk2::ShaderModuleInfo::ensureNoEmptyDescriptorTypes(shader_module_info.getDescriptorsTypes(VK_MAIN_BINDING_SET));
	std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_bindings = createDescriptorSetLayoutBindings(descriptor_types, vk::ShaderStageFlagBits::eCompute);
	vk::DescriptorSetLayoutBinding descriptor_set_layout_rassert_binding(VK_RASSERT_CODE_BINDING_SLOT, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);

	vk::raii::DescriptorSetLayout descriptor_set_layout_raii = vk->getDevice().createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, descriptor_set_layout_bindings));
	vk::raii::DescriptorSetLayout descriptor_set_layout_rassert_raii = vk->getDevice().createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, descriptor_set_layout_rassert_binding));
	std::vector<vk::DescriptorSetLayout> descriptor_set_layouts_non_raii = { descriptor_set_layout_raii, descriptor_set_layout_rassert_raii };
	vk::PipelineLayoutCreateInfo pipeline_create_info(vk::PipelineLayoutCreateFlags(), descriptor_set_layouts_non_raii);

	// see https://vkguide.dev/docs/new_chapter_2/vulkan_pushconstants/ and https://vkguide.dev/docs/chapter-3/push_constants/
	std::vector<vk::PushConstantRange> push_constant_ranges = { vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, shader_module_info.getPushConstantSize()) };
	pipeline_create_info.setPushConstantRanges(push_constant_ranges);

	vk::raii::PipelineLayout pipeline_layout = vk->getDevice().createPipelineLayout(pipeline_create_info);

	// TODO remove me?
//	vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, shader_module, name_.c_str());

	vk::ComputePipelineCreateInfo compute_pipeline_create_info;
	compute_pipeline_create_info.setLayout(pipeline_layout);
	compute_pipeline_create_info.setStage(pipeline_stages_create_info);

    // To make possible automatic subdivision of workload (i.e. to use dispatchBase(baseGroup, groupCount)),
    // i.e. to prevent validation error:
    // If any of baseGroupX (123456), baseGroupY (0), or baseGroupZ (0) are not zero, then the bound compute pipeline must have been created with the VK_PIPELINE_CREATE_DISPATCH_BASE flag
    compute_pipeline_create_info.setFlags(vk::PipelineCreateFlagBits::eDispatchBase);

	VulkanKernel *kernel = new VulkanKernel(compute_program->programName());
//		vk::raii::PipelineCache pipeline_cache = vk->getDevice().createPipelineCache(vk::PipelineCacheCreateInfo()); // TODO use pipeline caching
	vk::raii::Pipeline pipeline = vk->getDevice().createComputePipeline(nullptr, compute_pipeline_create_info);
	kernel->create(std::move(descriptor_set_layout_raii), std::move(descriptor_set_layout_rassert_raii), std::move(pipeline_layout), std::move(pipeline), std::move(shader_module_info));
	return kernel;
}

avk2::VulkanKernel *avk2::KernelSource::compileRasterizationKernel(const std::shared_ptr<VulkanEngine> &vk)
{
	rassert(false, 618525500); // TODO
}

avk2::VulkanKernel *avk2::KernelSource::getKernel(const std::shared_ptr<VulkanEngine> &vk)
{
	VulkanKernel *kernel = vk->findKernel(id_);
	if (kernel)
		return kernel;

	timer tm;
	tm.start();

	if (isCompute()) {
		kernel = compileComputeKernel(vk);
	} else {
		rassert(isRasterization(), 751588296);
		kernel = compileRasterizationKernel(vk);
	}

	double compilation_time = tm.elapsed();
	if (compilation_time > 0.1) {
		std::cout << "Vulkan kernels compilation done in " << compilation_time << " seconds for " << vk->device().name << std::endl;
	}

	vk->kernels()[id_] = kernel;
	return kernel;
}

bool avk2::KernelSource::parseArg(std::vector<avk2::KernelSource::Arg> &args, const Arg &arg)
{
	if (arg.is_null) {
		return false; // argument is empty - so we don't need to check other arguments
	}

	if (arg.buffer || arg.image) {
		args.push_back(arg);
	} else {
		rassert(arg.is_null && false, 453124124312312); // impossible case
	}
	return true;
}

std::vector<avk2::KernelSource::Arg> avk2::KernelSource::parseArgs(const Arg &arg0, const Arg &arg1, const Arg &arg2, const Arg &arg3, const Arg &arg4, const Arg &arg5, const Arg &arg6, const Arg &arg7, const Arg &arg8, const Arg &arg9, const Arg &arg10, const Arg &arg11, const Arg &arg12, const Arg &arg13, const Arg &arg14, const Arg &arg15, const Arg &arg16, const Arg &arg17, const Arg &arg18, const Arg &arg19, const Arg &arg20, const Arg &arg21, const Arg &arg22, const Arg &arg23, const Arg &arg24, const Arg &arg25, const Arg &arg26, const Arg &arg27, const Arg &arg28, const Arg &arg29, const Arg &arg30, const Arg &arg31, const Arg &arg32, const Arg &arg33, const Arg &arg34, const Arg &arg35, const Arg &arg36, const Arg &arg37, const Arg &arg38, const Arg &arg39, const Arg &arg40)
{
	std::vector<avk2::KernelSource::Arg> args;
	if (!parseArg(args, arg0))  return args;
	if (!parseArg(args, arg1))  return args;
	if (!parseArg(args, arg2))  return args;
	if (!parseArg(args, arg3))  return args;
	if (!parseArg(args, arg4))  return args;
	if (!parseArg(args, arg5))  return args;
	if (!parseArg(args, arg6))  return args;
	if (!parseArg(args, arg7))  return args;
	if (!parseArg(args, arg8))  return args;
	if (!parseArg(args, arg9))  return args;
	if (!parseArg(args, arg10)) return args;
	if (!parseArg(args, arg11)) return args;
	if (!parseArg(args, arg12)) return args;
	if (!parseArg(args, arg13)) return args;
	if (!parseArg(args, arg14)) return args;
	if (!parseArg(args, arg15)) return args;
	if (!parseArg(args, arg16)) return args;
	if (!parseArg(args, arg17)) return args;
	if (!parseArg(args, arg18)) return args;
	if (!parseArg(args, arg19)) return args;
	if (!parseArg(args, arg20)) return args;
	if (!parseArg(args, arg21)) return args;
	if (!parseArg(args, arg22)) return args;
	if (!parseArg(args, arg23)) return args;
	if (!parseArg(args, arg24)) return args;
	if (!parseArg(args, arg25)) return args;
	if (!parseArg(args, arg26)) return args;
	if (!parseArg(args, arg27)) return args;
	if (!parseArg(args, arg28)) return args;
	if (!parseArg(args, arg29)) return args;
	if (!parseArg(args, arg30)) return args;
	if (!parseArg(args, arg31)) return args;
	if (!parseArg(args, arg32)) return args;
	if (!parseArg(args, arg33)) return args;
	if (!parseArg(args, arg34)) return args;
	if (!parseArg(args, arg35)) return args;
	if (!parseArg(args, arg36)) return args;
	if (!parseArg(args, arg37)) return args;
	if (!parseArg(args, arg38)) return args;
	if (!parseArg(args, arg39)) return args;
	if (!parseArg(args, arg40)) return args;
	return args;
}

void avk2::KernelSource::exec(const PushConstant &params, const gpu::WorkSize &ws, const Arg &arg0, const Arg &arg1, const Arg &arg2, const Arg &arg3, const Arg &arg4, const Arg &arg5, const Arg &arg6, const Arg &arg7, const Arg &arg8, const Arg &arg9, const Arg &arg10, const Arg &arg11, const Arg &arg12, const Arg &arg13, const Arg &arg14, const Arg &arg15, const Arg &arg16, const Arg &arg17, const Arg &arg18, const Arg &arg19, const Arg &arg20, const Arg &arg21, const Arg &arg22, const Arg &arg23, const Arg &arg24, const Arg &arg25, const Arg &arg26, const Arg &arg27, const Arg &arg28, const Arg &arg29, const Arg &arg30, const Arg &arg31, const Arg &arg32, const Arg &arg33, const Arg &arg34, const Arg &arg35, const Arg &arg36, const Arg &arg37, const Arg &arg38, const Arg &arg39, const Arg &arg40)
{
	rassert(isCompute(), 983645706);

	timer total_t;
	total_t.start();

	gpu::Context context;

	VulkanKernel *kernel = getKernel(context.vk()); // constructing pipeline

	std::vector<Arg> args = parseArgs(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40);

	const PushConstant &push_constant = params;
	rassert(!push_constant.isEmpty(), 990402328745136); // this is not a strict requirement, but in all cases currently we use push constants, so let's ensure this

	std::vector<vk::DescriptorType> descriptor_types = kernel->getDescriptorTypes();
	vk::raii::DescriptorSet descriptor_set = context.vk()->allocateDescriptor(kernel->descriptorSetLayout(), descriptor_types);

	unsigned int nargs = args.size();
	std::vector<vk::DescriptorBufferInfo> buffers_info(nargs);
	std::vector<vk::DescriptorImageInfo> images_info(nargs);
	std::vector<vk::WriteDescriptorSet> descriptor_writes(nargs);
	for (size_t i = 0; i < nargs; ++i) {
		unsigned int binding = i;
		descriptor_writes[i] = vk::WriteDescriptorSet(descriptor_set, binding, 0, 1, descriptor_types[i]);
		if (args[i].buffer) {
			rassert(vk::DescriptorType::eStorageBuffer == descriptor_types[i], 963647016);

			// it is important to use vkWholeSize (i.e. buffer->size + buffer->nbytes_guard_suffix) for more sensitive out-of-bounds guard
			// because it seems that otherwise if buffer->size is used - at least NVIDIA driver prevents out-of-bounds access - but this masks the problem
			vk::DeviceSize buffer_size = vk::WholeSize;

			buffers_info[i] = vk::DescriptorBufferInfo(args[i].buffer->vkBufferData()->getBuffer(), args[i].buffer->vkoffset(), buffer_size);
			descriptor_writes[i].setBufferInfo(buffers_info[i]);
		} else if (args[i].image) {
			rassert(vk::DescriptorType::eStorageImage == descriptor_types[i] || vk::DescriptorType::eSampledImage == descriptor_types[i] || vk::DescriptorType::eCombinedImageSampler == descriptor_types[i], 423512341412);
			// TODO ADD LAYOUTS SUPPORT: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			vk::ImageView image_view;
			bool is_array = kernel->isImageArrayed(VK_MAIN_BINDING_SET, binding);
			if (!is_array) {
				rassert(args[i].image->cn() == 1, 712753733);
				image_view = args[i].image->vkImageData()->getImageChannelView(0, false);
				// this is to prevent validation error:
				// the descriptor (VkDescriptorSet 0x44349c0000000060[], binding 0, index 0) ImageView type is VK_IMAGE_VIEW_TYPE_2D_ARRAY but the OpTypeImage has (Dim = 2D) and (Arrayed = 0).
				// The Vulkan spec states: If a VkImageView is accessed as a result of this command, then the image view's viewType must match the Dim operand of the OpTypeImage as described in Instruction/Sampler/Image View Validation
			} else {
				image_view = args[i].image->vkImageData()->getImageView();
			}
			images_info[i] = vk::DescriptorImageInfo(args[i].image->vkImageData()->getSampler(), image_view, args[i].image->vkImageData()->getCurrentLayout());
			descriptor_writes[i].setImageInfo(images_info[i]);
		}
	}
	context.vk()->getDevice().updateDescriptorSets(descriptor_writes, nullptr);

	vk::raii::CommandBuffer command_buffer = context.vk()->createCommandBuffer();

	command_buffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
	command_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, kernel->pipeline());

	std::vector<vk::DescriptorSet> descriptor_sets_non_raii = { descriptor_set };
	std::shared_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout_rassert_raii;
	std::shared_ptr<vk::raii::DescriptorSet> rassert_descriptor_set;
	if (kernel->isRassertUsed()) {
		vk::DescriptorSetLayoutBinding descriptor_set_layout_rassert_binding(VK_RASSERT_CODE_BINDING_SLOT, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
		descriptor_set_layout_rassert_raii = std::make_shared<vk::raii::DescriptorSetLayout>(std::move(context.vk()->getDevice().createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, descriptor_set_layout_rassert_binding))));

		std::vector<vk::DescriptorType> rassert_descriptor_types(1, vk::DescriptorType::eStorageBuffer);
		rassert_descriptor_set = std::make_shared<vk::raii::DescriptorSet>(std::move(context.vk()->allocateDescriptor(*descriptor_set_layout_rassert_raii, rassert_descriptor_types)));

		vk::DescriptorBufferInfo buffer_info = vk::DescriptorBufferInfo(kernel->rassertCodeAndLineBuffer().vkBufferData()->getBuffer(), kernel->rassertCodeAndLineBuffer().vkoffset(), kernel->rassertCodeAndLineBuffer().size());
		vk::WriteDescriptorSet descriptor_write = vk::WriteDescriptorSet(*rassert_descriptor_set, VK_RASSERT_CODE_BINDING_SLOT, 0, 1, rassert_descriptor_types[0]);
		descriptor_write.setBufferInfo(buffer_info);

		context.vk()->getDevice().updateDescriptorSets(descriptor_write, nullptr);

		descriptor_sets_non_raii.push_back(*rassert_descriptor_set);
	}
	command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, vk::PipelineLayout(kernel->pipelineLayout()), 0, descriptor_sets_non_raii, nullptr);
 
	// we don't use C++ API here because we want to manually specify size of push constant (via passing push_constant.size())
	rassert(VKF.vkCmdPushConstants, 315723128637936);
	VKF.vkCmdPushConstants(vk::CommandBuffer(command_buffer), VkPipelineLayout(vk::PipelineLayout(kernel->pipelineLayout())), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant.size(), push_constant.ptr());

	std::vector<size_t> spirv_group_size = kernel->shaderModuleInfo().getGroupSize(name_);
	for (size_t d = 0; d < 3; ++d) {
		rassert(spirv_group_size[d] == ws.vkGroupSize()[d], 151466595490051);
	}

	dispatchAutoSubdivided(command_buffer, ws);

	command_buffer.end();
	last_exec_prepairing_time_ = total_t.elapsed();

	timer gpu_t;
	gpu_t.start();
	context.vk()->submitCommandBuffer(command_buffer);
	last_exec_gpu_time_ = gpu_t.elapsed();

	if (kernel->isRassertUsed()) {
		kernel->checkRassertCode();
	}

	if (context.isMemoryGuardsChecksAfterKernelsEnabled()) {
		for (size_t i = 0; i < nargs; ++i) {
			unsigned int binding = i;
			if (args[i].buffer) {
				rassert(args[i].buffer->checkMagicGuards(getProgramName() + "/binding=" + to_string(binding)), 371153662);
			}
		}
	}

	last_exec_total_time_ = total_t.elapsed();
}

void avk2::KernelSource::dispatchAutoSubdivided(const vk::raii::CommandBuffer &command_buffer, const gpu::WorkSize &ws) const {
#if 0
    // trivial implementation running the whole workload in once
    command_buffer.dispatch(ws.vkGroupCount()[0], ws.vkGroupCount()[1], ws.vkGroupCount()[2]);
#else
    // TODO if device supports larget work group count limits - it can be better to use them
    const size_t WORK_GROUP_COUNT_LIMITS[3] = {VK_MAX_COMPUTE_WORK_GROUP_COUNT_X, VK_MAX_COMPUTE_WORK_GROUP_COUNT_Y, VK_MAX_COMPUTE_WORK_GROUP_COUNT_Z};
    size_t passes_by_axis[3];
    size_t work_group_count_by_axis[3];
    for (size_t d = 0; d < 3; ++d) {
        passes_by_axis[d] = div_ceil(ws.vkGroupCount()[d], WORK_GROUP_COUNT_LIMITS[d]);
        work_group_count_by_axis[d] = div_ceil(ws.vkGroupCount()[d], passes_by_axis[d]);
    }

    for (size_t pass_z = 0; pass_z < passes_by_axis[2]; ++pass_z) {
        for (size_t pass_y = 0; pass_y < passes_by_axis[1]; ++pass_y) {
            for (size_t pass_x = 0; pass_x < passes_by_axis[0]; ++pass_x) {

                size_t pass_index[3] = {pass_x, pass_y, pass_z};
                size_t group_offset[3];
                size_t group_count[3];

                for (size_t d = 0; d < 3; ++d) {
                    size_t from = pass_index[d] * work_group_count_by_axis[d];
                    size_t to = std::min((pass_index[d] + 1) * work_group_count_by_axis[d], ws.vkGroupCount()[d]);
                    group_offset[d] = from;
                    group_count[d] = to - from;
                    rassert(group_count[d] <= WORK_GROUP_COUNT_LIMITS[d], 9234512354654362);
                }

                command_buffer.dispatchBase(group_offset[0], group_offset[1], group_offset[2],
                                            group_count[0], group_count[1], group_count[2]);
            }
        }
    }
#endif
}

avk2::RenderBuilder avk2::KernelSource::initRender(size_t width, size_t height)
{
	return RenderBuilder(*this, width, height);
}

void avk2::KernelSource::launchRender(const RenderBuilder &params, const Arg &arg0, const Arg &arg1, const Arg &arg2, const Arg &arg3, const Arg &arg4, const Arg &arg5, const Arg &arg6, const Arg &arg7, const Arg &arg8, const Arg &arg9, const Arg &arg10, const Arg &arg11, const Arg &arg12, const Arg &arg13, const Arg &arg14, const Arg &arg15, const Arg &arg16, const Arg &arg17, const Arg &arg18, const Arg &arg19, const Arg &arg20, const Arg &arg21, const Arg &arg22, const Arg &arg23, const Arg &arg24, const Arg &arg25, const Arg &arg26, const Arg &arg27, const Arg &arg28, const Arg &arg29, const Arg &arg30, const Arg &arg31, const Arg &arg32, const Arg &arg33, const Arg &arg34, const Arg &arg35, const Arg &arg36, const Arg &arg37, const Arg &arg38, const Arg &arg39, const Arg &arg40)
{
	rassert(isRasterization(), 850196408);

	timer total_t;
	total_t.start();

	gpu::Context context;

	// TODO: rework this:
	// getDescriptorTypes
	// descriptorSetLayout
	// VulkanKernel *kernel = getKernel(context.vk()); //	 constructing pipeline

	// TODO move these into compile Rasterization Kernel method
	std::vector<vk::PipelineShaderStageCreateInfo> pipeline_stages;
	std::vector<vk::DescriptorType> descriptor_types;
	std::shared_ptr<vk::raii::DescriptorSet> descriptor_set;
	std::shared_ptr<vk::raii::DescriptorSet> rassert_descriptor_set;
	std::shared_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout_raii;
	std::shared_ptr<vk::raii::DescriptorSetLayout> descriptor_set_layout_rassert_raii;
	bool is_rassert_used = false;
	gpu::gpu_mem_32u rassert_code_and_line_;
	std::shared_ptr<vk::raii::PipelineLayout> pipeline_layout;
	std::vector<std::shared_ptr<vk::raii::ShaderModule>> shader_modules;
	std::vector<avk2::ShaderModuleInfo> pipeline_stages_shader_module_info;
	{
		// ensuring that rasterization shaders are vert->frag or vert->tesc->tese->frag
		rassert(shaders_programs_.size() == 2 || shaders_programs_.size() == 4, 310992267);
		if (shaders_programs_.size() == 2) {
			rassert(shaders_programs_[0]->isProgramNameEndsWith("_vert"), 997669760); // a vertex shader
			rassert(shaders_programs_[1]->isProgramNameEndsWith("_frag"), 351354122); // a fragment shader
		} else if (shaders_programs_.size() == 4) {
			rassert(shaders_programs_[0]->isProgramNameEndsWith("_vert"), 997669566); // a vertex shader
			rassert(shaders_programs_[1]->isProgramNameEndsWith("_tesc"), 997664766); // a tessellation control shader
			rassert(shaders_programs_[2]->isProgramNameEndsWith("_tese"), 997663766); // a tessellation evaluation shader
			rassert(shaders_programs_[3]->isProgramNameEndsWith("_frag"), 351354128); // a fragment shader
		} else {
			rassert(false, 674998313);
		}

		for (auto program: shaders_programs_) {
			vk::ShaderStageFlagBits shader_stage;
				 if (program->isProgramNameEndsWith("_vert")) shader_stage = vk::ShaderStageFlagBits::eVertex;
			else if (program->isProgramNameEndsWith("_frag")) shader_stage = vk::ShaderStageFlagBits::eFragment;
			else if (program->isProgramNameEndsWith("_tesc")) shader_stage = vk::ShaderStageFlagBits::eTessellationControl;
			else if (program->isProgramNameEndsWith("_tese")) shader_stage = vk::ShaderStageFlagBits::eTessellationEvaluation;
			else rassert(false, program->programName(), 596279565);

			avk2::ShaderModuleInfo shader_module_info;
			std::shared_ptr<vk::raii::ShaderModule> shader_module = std::make_shared<vk::raii::ShaderModule>(std::move(createShaderModule(context.vk(), *program, &shader_module_info)));
			shader_modules.push_back(shader_module);

			vk::PipelineShaderStageCreateInfo pipeline_stages_create_info({}, shader_stage, *shader_module, name_.c_str());
			pipeline_stages.push_back(pipeline_stages_create_info);
			pipeline_stages_shader_module_info.push_back(shader_module_info);
		}

		{
			std::vector<vk::DescriptorSetLayout> descriptor_set_layouts_non_raii;

			avk2::sh_ptr_vk_engine vk = context.vk();
			std::set<unsigned int> descriptors_sets = avk2::ShaderModuleInfo::getMergedDescriptorsSets(pipeline_stages_shader_module_info);
			if (descriptors_sets.count(VK_MAIN_BINDING_SET) > 0) {
				descriptor_types = avk2::ShaderModuleInfo::getMergedDescriptorsTypes(pipeline_stages_shader_module_info, VK_MAIN_BINDING_SET);
				std::vector<vk::DescriptorSetLayoutBinding> descriptor_set_layout_bindings = createDescriptorSetLayoutBindings(descriptor_types, vk::ShaderStageFlagBits::eAllGraphics);
				descriptor_set_layout_raii = std::make_shared<vk::raii::DescriptorSetLayout>(std::move(vk->getDevice().createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, descriptor_set_layout_bindings))));
				descriptor_set = std::make_shared<vk::raii::DescriptorSet>(std::move(context.vk()->allocateDescriptor(*descriptor_set_layout_raii, descriptor_types)));
				descriptor_set_layouts_non_raii.push_back(*descriptor_set_layout_raii);
			} else if (descriptors_sets.count(VK_RASSERT_CODE_SET) > 0) {
				// we need to create empty descriptor at set#0 (i.e. VK_MAIN_BINDING_SET) so that next one will be set#1 (i.e. VK_RASSERT_CODE_SET)
				descriptor_set_layout_raii = std::make_shared<vk::raii::DescriptorSetLayout>(std::move(vk->getDevice().createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, 0))));
				descriptor_set_layouts_non_raii.push_back(*descriptor_set_layout_raii);
			}

			if (descriptors_sets.count(VK_RASSERT_CODE_SET) > 0) {
				// it measn that that common.vk (and so rassert.vk) was included from Vulkan kernel:
				// #include <libgpu/vulkan/vk/common.vk>
				vk::DescriptorSetLayoutBinding descriptor_set_layout_rassert_binding(VK_RASSERT_CODE_BINDING_SLOT, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAllGraphics);
				std::vector<vk::DescriptorType> rassert_descriptor_types = avk2::ShaderModuleInfo::getMergedDescriptorsTypes(pipeline_stages_shader_module_info, VK_RASSERT_CODE_SET);
				descriptor_set_layout_rassert_raii = std::make_shared<vk::raii::DescriptorSetLayout>(std::move(vk->getDevice().createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, descriptor_set_layout_rassert_binding))));
				rassert_descriptor_set = std::make_shared<vk::raii::DescriptorSet>(std::move(context.vk()->allocateDescriptor(*descriptor_set_layout_rassert_raii, rassert_descriptor_types)));
				is_rassert_used = avk2::ShaderModuleInfo::isDescriptorUsedInAny(pipeline_stages_shader_module_info, VK_RASSERT_CODE_SET, VK_RASSERT_CODE_BINDING_SLOT);
				if (is_rassert_used) {
					rassert_code_and_line_.resizeN(4);
					unsigned int data_code_and_line[4] = {VK_RASSERT_CODE_MAGIC_GUARDS,
														  VK_RASSERT_CODE_EMPTY, VK_RASSERT_LINE_EMPTY,
														  VK_RASSERT_CODE_MAGIC_GUARDS};
					rassert_code_and_line_.writeN(data_code_and_line, 4);

					rassert(rassert_descriptor_types.size() == 1, 85374765);
					rassert(rassert_descriptor_types[0] == vk::DescriptorType::eStorageBuffer, 132105684);

					vk::DescriptorBufferInfo buffer_info = vk::DescriptorBufferInfo(rassert_code_and_line_.vkBufferData()->getBuffer(), rassert_code_and_line_.vkoffset(), rassert_code_and_line_.size());
					vk::WriteDescriptorSet descriptor_write = vk::WriteDescriptorSet(*rassert_descriptor_set.get(), VK_RASSERT_CODE_BINDING_SLOT, 0, 1, rassert_descriptor_types[0]);
					descriptor_write.setBufferInfo(buffer_info);

					context.vk()->getDevice().updateDescriptorSets(descriptor_write, nullptr);
				}
				descriptor_set_layouts_non_raii.push_back(*descriptor_set_layout_rassert_raii);
			}

			vk::PipelineLayoutCreateInfo pipeline_create_info(vk::PipelineLayoutCreateFlags(), descriptor_set_layouts_non_raii);

			// see https://vkguide.dev/docs/new_chapter_2/vulkan_pushconstants/ and https://vkguide.dev/docs/chapter-3/push_constants/
			size_t push_constant_size = avk2::ShaderModuleInfo::getMergedPushConstantSize(pipeline_stages_shader_module_info);
			std::vector<vk::PushConstantRange> push_constant_ranges = { vk::PushConstantRange(vk::ShaderStageFlagBits::eAllGraphics, 0, push_constant_size) };
			if (push_constant_size > 0) {
				pipeline_create_info.setPushConstantRanges(push_constant_ranges);
			}

			pipeline_layout = std::make_shared<vk::raii::PipelineLayout>(std::move(vk->getDevice().createPipelineLayout(pipeline_create_info)));
		}
	}

	std::vector<Arg> args_uniforms = parseArgs(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40);

//	std::vector<vk::DescriptorType> descriptor_types = kernel->getDescriptorTypes();
//	vk::raii::DescriptorSet descriptor_set = context.vk()->allocateDescriptor(kernel->descriptorSetLayout(), descriptor_types);
	{
	unsigned int nargs = args_uniforms.size();
	std::vector<vk::DescriptorBufferInfo> buffers_info(nargs);
	std::vector<vk::DescriptorImageInfo> images_info(nargs);
	std::vector<vk::WriteDescriptorSet> descriptor_writes(nargs);
	rassert(nargs == descriptor_types.size(), 992024390169);
	for (size_t i = 0; i < nargs; ++i) {
		unsigned int binding = i;
		rassert(descriptor_set, 787763202);
		descriptor_writes[i] = vk::WriteDescriptorSet(*descriptor_set, binding, 0, 1, descriptor_types[i]);
		if (args_uniforms[i].buffer) {
			rassert(vk::DescriptorType::eStorageBuffer == descriptor_types[i], 9636470163);

			// it is important to use vkWholeSize (i.e. buffer->size + buffer->nbytes_guard_suffix) for more sensitive out-of-bounds guard
			// because it seems that otherwise if buffer->size is used - at least NVIDIA driver prevents out-of-bounds access - but this masks the problem
			vk::DeviceSize buffer_size = vk::WholeSize;

			buffers_info[i] = vk::DescriptorBufferInfo(args_uniforms[i].buffer->vkBufferData()->getBuffer(), args_uniforms[i].buffer->vkoffset(), buffer_size);
			descriptor_writes[i].setBufferInfo(buffers_info[i]);
		} else if (args_uniforms[i].image) {
			rassert(vk::DescriptorType::eStorageImage == descriptor_types[i] || vk::DescriptorType::eSampledImage == descriptor_types[i] || vk::DescriptorType::eCombinedImageSampler == descriptor_types[i], 4235123414123);
			// TODO ADD LAYOUTS SUPPORT: VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			vk::ImageView image_view;
			bool is_descriptor_used = false;
			bool is_array = false;
			for (auto &shader_module: pipeline_stages_shader_module_info) {
				if (shader_module.isDescriptorUsed(VK_MAIN_BINDING_SET, binding)) {
					if (!is_descriptor_used) {
						is_array = shader_module.isImageArrayed(VK_MAIN_BINDING_SET, binding);
					} else {
						rassert(is_array == shader_module.isImageArrayed(VK_MAIN_BINDING_SET, binding), 747456074);
					}
					is_descriptor_used = true;
				}
			}
			// TODO generalize with exec(...)
			if (!is_array) {
				rassert(args_uniforms[i].image->cn() == 1, 712753733);
				image_view = args_uniforms[i].image->vkImageData()->getImageChannelView(0, false);
				// this is to prevent validation error:
				// the descriptor (VkDescriptorSet 0x44349c0000000060[], binding 0, index 0) ImageView type is VK_IMAGE_VIEW_TYPE_2D_ARRAY but the OpTypeImage has (Dim = 2D) and (Arrayed = 0).
				// The Vulkan spec states: If a VkImageView is accessed as a result of this command, then the image view's viewType must match the Dim operand of the OpTypeImage as described in Instruction/Sampler/Image View Validation
			} else {
				image_view = args_uniforms[i].image->vkImageData()->getImageView();
			}
			images_info[i] = vk::DescriptorImageInfo(args_uniforms[i].image->vkImageData()->getSampler(), image_view, args_uniforms[i].image->vkImageData()->getCurrentLayout());
			descriptor_writes[i].setImageInfo(images_info[i]);
		} else {
			rassert(false, 758388889); // this argument is non-empty and is uniform, so it is either buffer or image
		}
	}
	context.vk()->getDevice().updateDescriptorSets(descriptor_writes, nullptr);
	}

	std::shared_ptr<vk::raii::RenderPass> render_pass;
	{
		std::vector<vk::AttachmentDescription> attachments_description;
		std::vector<vk::AttachmentReference> color_attachment_references;
		std::shared_ptr<vk::AttachmentReference> depth_stencil_attachment;
		size_t attachment_slot = 0;
		for (const avk2::ImageAttachment &arg_attachment: params.depth_and_color_attachments_) {
			const gpu::shared_device_image &image = arg_attachment.getImage();
			for (size_t c = 0; c < image.cn(); ++c) {
				vk::Format format;
				if (arg_attachment.isColorAttachment()) {
					format = context.vk()->device().typeToVkFormat(image.dataType());
				} else {
					rassert(arg_attachment.isDepthStencilAttachment(), 45612346132);
					format = context.vk()->device().typeToVkDepthStencilFormat(image.dataType());
				}

				vk::AttachmentLoadOp load_op = vk::AttachmentLoadOp::eLoad; // TODO try to improve performance with vk::AttachmentLoadOp::eDontCare where possible
				if (arg_attachment.hasClearValue()) {
					load_op = vk::AttachmentLoadOp::eClear; // note that attachments will be cleared only in render_area
				}

				vk::ImageLayout initial_layout = vk::ImageLayout::eGeneral;
				vk::ImageLayout final_layout = vk::ImageLayout::eGeneral;

				// TODO try to improve performance with non-general layout, but it seems to be not important (and even have a negative effect) on NVIDIA and AMD desktop GPUs:
				// - https://www.reddit.com/r/vulkan/comments/1b72me6/comment/ktfqczw
				// - https://www.reddit.com/r/vulkan/comments/1b72me6/comment/ktl6wm6
				// - https://www.reddit.com/r/vulkan/comments/104hmyx/comment/j38rlp1
	//			if (attachment.isDepthStencilAttachment()) {
	//				initial_layout = final_layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
	//			} else {
	//				initial_layout = final_layout = vk::ImageLayout::eColorAttachmentOptimal;
	//			}

				attachments_description.push_back(vk::AttachmentDescription({}, format, vk::SampleCountFlagBits::e1,
																			load_op, vk::AttachmentStoreOp::eStore, load_op, vk::AttachmentStoreOp::eStore, // TODO try to improve performance with vk::AttachmentStoreOp::eDontCare where possible
																			initial_layout, final_layout));
				arg_attachment.getImage().vkImageData()->updateLayout(initial_layout, final_layout);
				if (arg_attachment.isColorAttachment()) {
					color_attachment_references.push_back(vk::AttachmentReference(attachment_slot, final_layout));
				} else {
					rassert(arg_attachment.isDepthStencilAttachment(), 167326983);
					rassert(!depth_stencil_attachment, 236398326); // checking that depth attachment is unique
					depth_stencil_attachment = std::shared_ptr<vk::AttachmentReference>(new vk::AttachmentReference(attachment_slot, final_layout));
				}
				++attachment_slot;
			}
			size_t nattachments = attachment_slot;
			rassert(nattachments <= VK_MAX_FRAGMENT_OUTPUT_ATTACHMENTS_USED, nattachments, 736698915);
		}

		// see https://github.com/SaschaWillems/Vulkan/blob/9756ad8c2368e0b94f149a9dc20a47b253feceff/examples/triangle/triangle.cpp#L582
		// see about input attachments (they are limited to pixel-local access so we don't use them) - https://www.saschawillems.de/blog/2018/07/19/vulkan-input-attachments-and-sub-passes
		vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, color_attachment_references, {}, depth_stencil_attachment.get(), {});
		vk::RenderPassCreateInfo render_pass_create_info({}, attachments_description, subpass, {});
		render_pass = std::shared_ptr<vk::raii::RenderPass>(new vk::raii::RenderPass(context.vk()->getDevice(), render_pass_create_info));
	}

	std::shared_ptr<vk::raii::Framebuffer> framebuffer;
	{
		std::vector<vk::ImageView> attachments_image_views;
		uint32_t nlayers = 0;
		for (const avk2::ImageAttachment &arg_attachment: params.depth_and_color_attachments_) {
			const gpu::shared_device_image &image = arg_attachment.getImage();
			for (size_t c = 0; c < image.cn(); ++c) {
				attachments_image_views.push_back(arg_attachment.getImage().vkImageData()->getImageChannelView(c));
				rassert(arg_attachment.getImage().width() >= params.viewport_width_ && arg_attachment.getImage().height() >= params.viewport_height_, 131106553);
				nlayers = 1;
			}
		}
		rassert(nlayers >= 1, 488150461);
		vk::FramebufferCreateInfo framebuffer_create_info({}, *render_pass, attachments_image_views, params.viewport_width_, params.viewport_height_, nlayers);
		framebuffer = std::shared_ptr<vk::raii::Framebuffer>(new vk::raii::Framebuffer(context.vk()->getDevice(), framebuffer_create_info));
	}

	// see https://github.com/KhronosGroup/Vulkan-Hpp/blob/main/vk_raii_ProgrammingGuide.md#15-drawing-a-cube
	std::shared_ptr<vk::raii::Pipeline> graphics_pipeline;
	{
		std::vector<vk::DynamicState> dynamic_states;

		std::vector<vk::VertexInputBindingDescription> vertex_binding_descriptions = { params.geometry_vertices_.buildBindingDescription() };
		std::vector<vk::VertexInputAttributeDescription> vertex_attribute_descriptions = params.geometry_vertices_.buildAttributeDescriptions();
		vk::PipelineVertexInputStateCreateInfo pipeline_vertex_input({}, vertex_binding_descriptions, vertex_attribute_descriptions);

		// see https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#drawing-triangle-lists-with-adjacency
		vk::PrimitiveTopology primitive_topology = params.faces_with_adjacency_ ? vk::PrimitiveTopology::eTriangleListWithAdjacency : vk::PrimitiveTopology::eTriangleList;
		vk::PipelineInputAssemblyStateCreateInfo pipeline_input_assembly({}, primitive_topology);

		dynamic_states.push_back(vk::DynamicState::eViewport);
		dynamic_states.push_back(vk::DynamicState::eScissor);
		vk::PipelineViewportStateCreateInfo pipeline_viewport({}, 1, nullptr, 1, nullptr); // we don't specify viewport and scissor because they are specified dynamically

		vk::PipelineRasterizationStateCreateInfo pipeline_rasterization({}, vk::False, vk::False, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone);
		if (params.polygon_mode_wireframe_) {
			pipeline_rasterization.polygonMode = vk::PolygonMode::eLine;
		}
		pipeline_rasterization.lineWidth = 1.0f; // to fix Validation Error: [ VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00749 ] | ... | vkCreateGraphicsPipelines(): pCreateInfos[0].pRasterizationState->lineWidth is 0.000000, but the line width state is static (pCreateInfos[0].pDynamicState->pDynamicStates does not contain VK_DYNAMIC_STATE_LINE_WIDTH) and wideLines feature was not enabled. The Vulkan spec states: If the pipeline requires pre-rasterization shader state, and the wideLines feature is not enabled, and no element of the pDynamicStates member of pDynamicState is VK_DYNAMIC_STATE_LINE_WIDTH, the lineWidth member of pRasterizationState must be 1.0 (https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00749)
		vk::PipelineMultisampleStateCreateInfo pipeline_multisample({}, vk::SampleCountFlagBits::e1);
		vk::PipelineDepthStencilStateCreateInfo pipeline_depth_stencil({}, vk::True, params.depth_write_enable_, vk::CompareOp::eLessOrEqual, vk::False, vk::False);

		// Color blend state describes how blend factors are calculated (if used)
		// We need one blend attachment state per color attachment (even if blending is not used)
		std::vector<vk::PipelineColorBlendAttachmentState> color_blend_attachments;
		for (const avk2::ImageAttachment &arg_attachment: params.depth_and_color_attachments_) {
			const gpu::shared_device_image &image = arg_attachment.getImage();
			for (size_t c = 0; c < image.cn(); ++c) {
				if (arg_attachment.isColorAttachment()) {
					vk::PipelineColorBlendAttachmentState color_blend_attachment_state(vk::False);
					if (params.color_attachments_blending_enabled_) {
						// see https://vulkan-tutorial.com/Drawing_a_triangle/Graphics_pipeline_basics/Fixed_functions#page_Color-blending
						// finalColor.rgb = (srcColorBlendFactor * newColor.rgb) <colorBlendOp> (dstColorBlendFactor * oldColor.rgb);
						// and in our case it is trivial:
						// finalColor.r = (1 * newColor.r) + (1 * oldColor.r);
						color_blend_attachment_state.setBlendEnable(vk::True);
						color_blend_attachment_state.setSrcColorBlendFactor(vk::BlendFactor::eOne);
						color_blend_attachment_state.setDstColorBlendFactor(vk::BlendFactor::eOne);
						color_blend_attachment_state.setColorBlendOp(vk::BlendOp::eAdd);
						// let's explicitly state that we don't use alpha - because we create separate image attachment per channel:
						color_blend_attachment_state.setSrcAlphaBlendFactor(vk::BlendFactor::eZero);
						color_blend_attachment_state.setDstAlphaBlendFactor(vk::BlendFactor::eZero);
						color_blend_attachment_state.setAlphaBlendOp(vk::BlendOp::eAdd);
					}
					vk::ColorComponentFlags single_channel = vk::ColorComponentFlagBits::eR;
					color_blend_attachment_state.setColorWriteMask(single_channel);
					color_blend_attachments.push_back(color_blend_attachment_state);
				}
			}
		}
		if (params.color_attachments_blending_enabled_) {
			rassert(color_blend_attachments.size() > 0, 221107316);
		}
		vk::PipelineColorBlendStateCreateInfo pipeline_color_blend;
		pipeline_color_blend.setAttachments(color_blend_attachments);

		// pipeline_color_blend.setAttachments(); // TODO specify color blending if needed
		vk::PipelineDynamicStateCreateInfo pipeline_dynamic({}, dynamic_states);

		vk::GraphicsPipelineCreateInfo graphics_pipeline_create_info(
			{},
			pipeline_stages, // TODO create on per-shader stage basis?
			&pipeline_vertex_input,
			&pipeline_input_assembly,
			nullptr, // pTessellationState
			&pipeline_viewport,
			&pipeline_rasterization,
			&pipeline_multisample,
			&pipeline_depth_stencil,
			&pipeline_color_blend,
			&pipeline_dynamic,
			*pipeline_layout, // TODO: rework this
			vk::RenderPass(*render_pass)
		);

		// TODO it should be cached (assuming any property haven't changed)
		vk::Optional<const vk::raii::PipelineCache> no_pipeline_cache = nullptr; // TODO we can try to use persistent vk::raii::PipelineCache for faster first rendering - can be important for multiprocess cluster processing
		graphics_pipeline = std::shared_ptr<vk::raii::Pipeline>(new vk::raii::Pipeline(context.vk()->getDevice(), no_pipeline_cache, graphics_pipeline_create_info));
	}

	{
		vk::raii::CommandBuffer command_buffer = context.vk()->createCommandBuffer();

		command_buffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
		vk::Rect2D render_area(vk::Offset2D(0, 0), vk::Extent2D(params.viewport_width_, params.viewport_height_));
		std::vector<vk::ClearValue> clear_values;

		// w.r.t. specification https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPassBeginInfo.html
		// "Only elements corresponding to cleared attachments are used. Other elements of pClearValues are ignored."
		// so we can use any value - it will be ignored
		const float ignored_clear_color_value = std::numeric_limits<float>::max();

		for (const avk2::ImageAttachment &arg_attachment: params.depth_and_color_attachments_) {
			const gpu::shared_device_image &image = arg_attachment.getImage();
			if (arg_attachment.isColorAttachment()) {
				for (size_t c = 0; c < image.cn(); ++c) {
					if (arg_attachment.hasClearValue()) {
						clear_values.push_back(arg_attachment.getChannelColorClearValue(c)); // note that attachments will be cleared only in render_area
					} else {
						clear_values.push_back(vk::ClearColorValue(ignored_clear_color_value, ignored_clear_color_value, ignored_clear_color_value, ignored_clear_color_value)); // this clear value will be ignored
					}
				}
			} else {
				rassert(arg_attachment.isDepthStencilAttachment(), 440119260);
				if (arg_attachment.hasClearValue()) {
					clear_values.push_back(arg_attachment.getClearValue()); // note that attachments will be cleared only in render_area
				} else {
					clear_values.push_back(vk::ClearValue(ignored_clear_color_value)); // this clear value will be ignored
				}
			}
		}
		vk::RenderPassBeginInfo render_pass_begin_info(*render_pass, *framebuffer, render_area, clear_values);
		command_buffer.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);
		command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_pipeline);

		std::vector<vk::DescriptorSet> descriptor_sets_non_raii;
		unsigned int first_set = std::numeric_limits<unsigned int>::max();
		if (descriptor_set) {
			first_set = std::min(first_set, (unsigned int) VK_MAIN_BINDING_SET);
			descriptor_sets_non_raii.push_back(*descriptor_set);
		}
		if (rassert_descriptor_set) {
			first_set = std::min(first_set, (unsigned int) VK_RASSERT_CODE_SET);
			descriptor_sets_non_raii.push_back(*rassert_descriptor_set);
		}
		if (descriptor_sets_non_raii.size() > 0) {
			command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, first_set, descriptor_sets_non_raii, nullptr);
		}
 
		// we don't use C++ API here because we want to manually specify size of push constant (via passing push_constant_.size())
		if (params.push_constant_.size() > 0) {
			rassert(VKF.vkCmdPushConstants, 3157231286379363);
			VKF.vkCmdPushConstants(vk::CommandBuffer(command_buffer), VkPipelineLayout(**pipeline_layout), VK_SHADER_STAGE_ALL_GRAPHICS, 0, params.push_constant_.size(), params.push_constant_.ptr());
		}

		rassert(params.viewport_min_depth_ >= 0.0f && params.viewport_max_depth_ <= 1.0f, 691689354);
		// note that sadly we can't use viewport depth with [0-MAX_FLT] range due to absence of support for VK_EXT_depth_range_unrestricted on MacOS
		// see https://github.com/KhronosGroup/MoltenVK/issues/1576 and https://vulkan.gpuinfo.org/listdevicescoverage.php?platform=macos&extension=VK_EXT_depth_range_unrestricted
		// also it is not supported on Intel Arc, f.e. on Intel A580 - https://vulkan.gpuinfo.org/displayreport.php?id=27375#device
		command_buffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
												   params.viewport_width_, params.viewport_height_,
												   params.viewport_min_depth_, params.viewport_max_depth_));
		command_buffer.setScissor(0, render_area);

		command_buffer.bindVertexBuffers(VK_VERTICES_BINDING, params.geometry_vertices_.buffer(), params.geometry_vertices_.offset());
		command_buffer.bindIndexBuffer(params.geometry_indices_.buffer(), params.geometry_indices_.offset(), vk::IndexType::eUint32); // TODO use offset to render with subdivision (to prevent TDR timeout)

		size_t first_face = 0; // TODO use offset to render with subdivision (to prevent TDR timeout)
		size_t first_index = 3 * first_face;
		command_buffer.drawIndexed(params.geometry_indices_.nindices(), 1, first_index, 0, 0);

		command_buffer.endRenderPass();
		command_buffer.end();

		last_exec_prepairing_time_ = total_t.elapsed();

		timer gpu_t;
		gpu_t.start();
		context.vk()->submitCommandBuffer(command_buffer);
		last_exec_gpu_time_ = gpu_t.elapsed();
	}

	if (is_rassert_used) {
		// TODO will be in kernel->checkRassertCode();
		unsigned int data_code_and_line[4] = {0, 239, 17, 0};
		rassert_code_and_line_.readN(data_code_and_line, 4);

		// if there is out-of-bounds writing in some kernels - they can accidentally write their trash values into code/line memory bytes
		// so let's check that magic guards were untouched (if there are out-of-bounds writing - it will probably overwrite magic guards too)
		rassert(data_code_and_line[0] == VK_RASSERT_CODE_MAGIC_GUARDS, 34521253541);
		rassert(data_code_and_line[3] == VK_RASSERT_CODE_MAGIC_GUARDS, 1535346445);

		unsigned int code = data_code_and_line[1];
		unsigned int line = data_code_and_line[2];
		rassert_gpu(code == VK_RASSERT_CODE_EMPTY && line == VK_RASSERT_LINE_EMPTY, "Vulkan kernel detected error on GPU with code=" + to_string(code) + " at kernel line=" + to_string(line));
	}

	if (context.isMemoryGuardsChecksAfterKernelsEnabled()) {
		for (size_t i = 0; i < args_uniforms.size(); ++i) {
			unsigned int binding = i;
			if (args_uniforms[i].buffer) {
				rassert(args_uniforms[i].buffer->checkMagicGuards(getProgramName() + "/binding=" + to_string(binding)), 68607859);
			}
		}
	}

	last_exec_total_time_ = total_t.elapsed();
}

avk2::VulkanKernelArg::VulkanKernelArg(const gpu::shared_device_buffer &arg)
{
	init();
	buffer = &arg;
}

avk2::VulkanKernelArg::VulkanKernelArg(const gpu::shared_device_image &arg)
{
	init();
	image = &arg;
}

avk2::ImageAttachment::ImageAttachment(gpu::shared_device_image &image, bool is_color)												: image_(image), is_color_(is_color) {}
avk2::ImageAttachment::ImageAttachment(gpu::shared_device_image &image, point4<float>        clear_color_value)						: image_(image), clear_value_(std::shared_ptr<vk::ClearValue>(new vk::ClearValue(vk::ClearColorValue(clear_color_value[0], clear_color_value[1], clear_color_value[2], clear_color_value[3])))), is_color_(true) {}
avk2::ImageAttachment::ImageAttachment(gpu::shared_device_image &image, point4<int>          clear_color_value)						: image_(image), clear_value_(std::shared_ptr<vk::ClearValue>(new vk::ClearValue(vk::ClearColorValue(clear_color_value[0], clear_color_value[1], clear_color_value[2], clear_color_value[3])))), is_color_(true) {}
avk2::ImageAttachment::ImageAttachment(gpu::shared_device_image &image, point4<unsigned int> clear_color_value)						: image_(image), clear_value_(std::shared_ptr<vk::ClearValue>(new vk::ClearValue(vk::ClearColorValue(clear_color_value[0], clear_color_value[1], clear_color_value[2], clear_color_value[3])))), is_color_(true) {}
avk2::ImageAttachment::ImageAttachment(gpu::shared_device_image &image, float clear_depth_value, unsigned int clear_stencil_value)	: image_(image), clear_value_(std::shared_ptr<vk::ClearValue>(new vk::ClearValue(vk::ClearDepthStencilValue(clear_depth_value, clear_stencil_value)))), is_color_(false) {}

vk::ClearValue avk2::ImageAttachment::getChannelColorClearValue(unsigned int c) const
{
	rassert(hasClearValue(), 311187706);
	rassert(isColorAttachment(), 373847281);
	rassert(c < getImage().cn(), 512865843);
	rassert(c < 4, 606933658);
	float value = (*clear_value_).color.float32[c];
	return vk::ClearValue(vk::ClearColorValue(value, 0.0f, 0.0f, 0.0f));
}

vk::VertexInputBindingDescription avk2::VertexInput::buildBindingDescription() const
{
	return vk::VertexInputBindingDescription(VK_VERTICES_BINDING, stride_, vk::VertexInputRate::eVertex);
}

std::vector<vk::VertexInputAttributeDescription> avk2::VertexInput::buildAttributeDescriptions() const
{
	gpu::Context context;
	rassert(context.type() == gpu::Context::TypeVulkan, 719991143);
	sh_ptr_vk_engine vk = context.vk();
	rassert(vk, 149287335);

	std::vector<vk::VertexInputAttributeDescription> descriptions;

	unsigned int vertex_attribute_location = 0;
	unsigned int attribute_offset = 0;

	if (attribute_ndim_ > 0) {
		descriptions.push_back(vk::VertexInputAttributeDescription(vertex_attribute_location++, VK_VERTICES_BINDING,
																   vk->device().typeToVkFormat(DataType32f, attribute_ndim_), attribute_offset));
		attribute_offset += attribute_ndim_ * sizeof(float);
	}

	if (attribute_nfloat_ > 0) {
		descriptions.push_back(vk::VertexInputAttributeDescription(vertex_attribute_location++, VK_VERTICES_BINDING,
																   vk->device().typeToVkFormat(DataType32f, attribute_nfloat_), attribute_offset));
		attribute_offset += attribute_nfloat_ * sizeof(float);
	}

	if (attribute_nuint_ > 0) {
		descriptions.push_back(vk::VertexInputAttributeDescription(vertex_attribute_location++, VK_VERTICES_BINDING,
																   vk->device().typeToVkFormat(DataType32u, attribute_nuint_), attribute_offset));
		attribute_offset += attribute_nuint_ * sizeof(unsigned int);
	}

	rassert(attribute_offset == stride_, 704555959);

	return descriptions;
}

vk::Buffer avk2::VertexInput::buffer() const
{
	return buffer_.vkBufferData()->getBuffer();
}

vk::Buffer avk2::IndicesInput::buffer() const
{
	return buffer_.vkBufferData()->getBuffer();
}

void avk2::VulkanKernelArg::init()
{
	is_null = false;
	buffer = nullptr;
	image = nullptr;
}

avk2::VulkanKernel::~VulkanKernel()
{
	descriptor_set_layout_.reset();
	descriptor_set_layout_rassert_.reset();
	pipeline_layout_.reset();
	pipeline_.reset();
	this->shader_module_info_.reset();
}

void avk2::VulkanKernel::create(vk::raii::DescriptorSetLayout &&descriptor_set_layout, vk::raii::DescriptorSetLayout &&descriptor_set_layout_for_rassert, vk::raii::PipelineLayout &&pipeline_layout, vk::raii::Pipeline &&pipeline, avk2::ShaderModuleInfo &&shader_module_info)
{
	this->descriptor_set_layout_ = std::make_shared<vk::raii::DescriptorSetLayout>(std::move(descriptor_set_layout));
	this->descriptor_set_layout_rassert_ = std::make_shared<vk::raii::DescriptorSetLayout>(std::move(descriptor_set_layout_for_rassert));
	this->pipeline_layout_ = std::make_shared<vk::raii::PipelineLayout>(std::move(pipeline_layout));
	this->pipeline_ = std::make_shared<vk::raii::Pipeline>(std::move(pipeline));
	this->shader_module_info_ = std::make_shared<avk2::ShaderModuleInfo>(std::move(shader_module_info));
	descriptor_types_ = avk2::ShaderModuleInfo::ensureNoEmptyDescriptorTypes(this->shader_module_info_->getDescriptorsTypes(VK_MAIN_BINDING_SET));
	is_rassert_used_ = this->shader_module_info_->isDescriptorUsed(VK_RASSERT_CODE_SET, VK_RASSERT_CODE_BINDING_SLOT);

	if (is_rassert_used_) {
		gpu::Context context;
		rassert(context.type() == gpu::Context::TypeVulkan, 483464293);

		rassert_code_and_line_.resizeN(4);
		unsigned int data_code_and_line[4] = {VK_RASSERT_CODE_MAGIC_GUARDS,
											  VK_RASSERT_CODE_EMPTY, VK_RASSERT_LINE_EMPTY,
											  VK_RASSERT_CODE_MAGIC_GUARDS}; 
		rassert_code_and_line_.writeN(data_code_and_line, 4);
	}
}

bool avk2::VulkanKernel::isImageArrayed(unsigned int set, unsigned int binding)
{
	return shader_module_info_->isImageArrayed(set, binding);
}

void avk2::VulkanKernel::checkRassertCode()
{
	rassert(isRassertUsed(), 683610370);
	unsigned int data_code_and_line[4] = {0, 239, 17, 0};
	rassert_code_and_line_.readN(data_code_and_line, 4);

	// if there is out-of-bounds writing in some kernels - they can accidentally write their trash values into code/line memory bytes
	// so let's check that magic guards were untouched (if there are out-of-bounds writing - it will probably overwrite magic guards too)
	rassert(data_code_and_line[0] == VK_RASSERT_CODE_MAGIC_GUARDS, 873958803);
	rassert(data_code_and_line[3] == VK_RASSERT_CODE_MAGIC_GUARDS, 453451234);

	unsigned int code = data_code_and_line[1];
	unsigned int line = data_code_and_line[2];
	rassert_gpu(code == VK_RASSERT_CODE_EMPTY && line == VK_RASSERT_LINE_EMPTY, "Vulkan kernel detected error on GPU with code=" + to_string(code) + " at kernel line=" + to_string(line));
}

avk2::RenderBuilder::RenderBuilder(KernelSource &kernel, size_t width, size_t height)
: kernel_(kernel), viewport_width_(width), viewport_height_(height)
{
	viewport_min_depth_		= 0.0;
	viewport_max_depth_		= 1.0f;
	faces_with_adjacency_	= false;
	polygon_mode_wireframe_	= false;
	depth_write_enable_		= true;
	color_attachments_blending_enabled_	= false;
}

avk2::RenderBuilder &avk2::RenderBuilder::geometryHelper(const avk2::VertexInput &vertices, const avk2::IndicesInput &faces)
{
	rassert(geometry_vertices_.isNull(), 638681160); // check for uniqueness of builder's geometry(...) call
	rassert(geometry_indices_.isNull(), 638681161); // check for uniqueness of builder's geometry(...) call
	geometry_vertices_ = vertices;
	geometry_indices_ = faces;
	return *this;
}

// depth framebuffer attachment will not be cleared (will be used as is - i.e. with vk::AttachmentLoadOp::eLoad)
avk2::RenderBuilder &avk2::RenderBuilder::setDepthAttachment(gpu::shared_device_depth_image &depth_buffer, bool depth_write_enable)
{
	depth_write_enable_ = depth_write_enable;
	return addDepthAttachmentHelper(avk2::DepthAttachment(depth_buffer));
}

// depth framebuffer attachment will be used with clear-on-load values for depth and stencil specified
// in most cases 1.0 should be used for depth, because depth range is [0; 1] (due to low adoption of VK_EXT_depth_range_unrestricted extension at the moment)
// note that depth framebuffer is optional
avk2::RenderBuilder &avk2::RenderBuilder::setDepthAttachment(gpu::shared_device_depth_image &depth_buffer, float clear_depth_value, unsigned int clear_stencil_value)
{
	return addDepthAttachmentHelper(avk2::DepthAttachment(depth_buffer, clear_depth_value, clear_stencil_value));
}

avk2::RenderBuilder &avk2::RenderBuilder::addDepthAttachmentHelper(const avk2::DepthAttachment &depth_attachment)
{
	rassert(depth_and_color_attachments_.size() == 0, 925300184); // checking that depth framebuffer attachment is in the zero slot (this is not mandatory, but probably it leads to more consistent code)
	depth_and_color_attachments_.push_back(depth_attachment);
	return *this;
}

// this image attachment will not be cleared (will be used as is - i.e. with vk::AttachmentLoadOp::eLoad)
// note that there can be any number of attachments - from zero and up to device/driver specific limit
avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image &color_attachment)
{
	return addColorAttachmentHelper(avk2::ImageAttachment(color_attachment, true));
}

template <typename T>
avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image_typed<T> &color_attachment, const T &single_channel_value)
{
	rassert(color_attachment.cn() == 1, 388381945);
	return addAttachment(color_attachment, {single_channel_value, 0, 0, 0});
}

// this image attachment will be used with clear-on-load values
// note that there can be any number of attachments - from zero and up to device/driver specific limit
template <typename T>
avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image_typed<T> &color_attachment, const point4<T> &clear_value)
{
	return addColorAttachmentHelper(avk2::ImageAttachment(color_attachment, clear_value));
}

avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image_typed<unsigned char> &color_attachment, const point4uc &clear_value)
{
	point4f clear_value_f32;
	for (int i = 0; i < 4; ++i) {
		clear_value_f32[i] = clear_value[i] / 255.0f;
	}
	return addColorAttachmentHelper(avk2::ImageAttachment(color_attachment, clear_value_f32));
}

avk2::RenderBuilder &avk2::RenderBuilder::addColorAttachmentHelper(const avk2::ImageAttachment &color_attachment)
{
	depth_and_color_attachments_.push_back(color_attachment);
	return *this;
}

void avk2::RenderBuilder::exec(const Arg &arg0, const Arg &arg1, const Arg &arg2, const Arg &arg3, const Arg &arg4, const Arg &arg5, const Arg &arg6, const Arg &arg7, const Arg &arg8, const Arg &arg9, const Arg &arg10, const Arg &arg11, const Arg &arg12, const Arg &arg13, const Arg &arg14, const Arg &arg15, const Arg &arg16, const Arg &arg17, const Arg &arg18, const Arg &arg19, const Arg &arg20, const Arg &arg21, const Arg &arg22, const Arg &arg23, const Arg &arg24, const Arg &arg25, const Arg &arg26, const Arg &arg27, const Arg &arg28, const Arg &arg29, const Arg &arg30, const Arg &arg31, const Arg &arg32, const Arg &arg33, const Arg &arg34, const Arg &arg35, const Arg &arg36, const Arg &arg37, const Arg &arg38, const Arg &arg39, const Arg &arg40)
{
	rassert(!geometry_vertices_.isNull(), 218981433); // geometry is mandatory
	rassert(!geometry_indices_.isNull(), 218981434); // geometry is mandatory

	// depth framebuffer and attachment images are optional (there can be any number of them - from zero and up to device/driver limit)
	// push constant is also optional

	kernel_.launchRender(*this, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, arg18, arg19, arg20, arg21, arg22, arg23, arg24, arg25, arg26, arg27, arg28, arg29, arg30, arg31, arg32, arg33, arg34, arg35, arg36, arg37, arg38, arg39, arg40);
}

void avk2::KernelsProfiler::addKernelProfilingTimes(const avk2::KernelSource &kernel)
{
	if (accumulate_per_kernel_times_) {
		per_kernel_total_sum_by_name_[kernel.getProgramName()] += kernel.getLastExecTotalTime();
	}
	kernels_total_		+= kernel.getLastExecTotalTime();
	kernels_prepairing_	+= kernel.getLastExecPrepairingTime();
	kernels_gpu_		+= kernel.getLastExecGPUTime();
}

void avk2::KernelsProfiler::printSlowestKernels(size_t top_k) const
{
	rassert(accumulate_per_kernel_times_, 785415761);
	std::vector<std::pair<double, std::string>> kernel_names_by_time;
	for (auto &it: per_kernel_total_sum_by_name_) {
		kernel_names_by_time.push_back(std::pair(it.second, it.first));
	}
	std::sort(kernel_names_by_time.begin(), kernel_names_by_time.end(), std::greater<>());
	std::cout << "slowest kernels:" << std::endl;
	for (size_t i = 0; i < std::min(kernel_names_by_time.size(), (size_t) 5); ++i) {
		std::cout << " - " << kernel_names_by_time[i].first << " sec - " << kernel_names_by_time[i].second << std::endl;
	}
}

template avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image_typed<unsigned int>	&color_attachment, const unsigned int	&single_channel_value);
template avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image_typed<int>			&color_attachment, const int			&single_channel_value);
template avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image_typed<float>			&color_attachment, const float			&single_channel_value);

template avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image_typed<unsigned int>	&color_attachment, const point4<unsigned int>	&clear_value);
template avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image_typed<int>			&color_attachment, const point4<int>			&clear_value);
template avk2::RenderBuilder &avk2::RenderBuilder::addAttachment(gpu::shared_device_image_typed<float>			&color_attachment, const point4<float>			&clear_value);
