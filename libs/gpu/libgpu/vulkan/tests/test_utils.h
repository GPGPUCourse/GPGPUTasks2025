#pragma once

#include <random>

#include <libgpu/context.h>
#include <libgpu/vulkan/engine.h>
#include <libbase/gtest_utils.h>
#include <libbase/runtime_assert.h>

// https://stackoverflow.com/a/61968208
#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
	EXPECT_GE((VAL), (MIN));           \
	EXPECT_LE((VAL), (MAX))

namespace {

std::vector<gpu::Device> enumVKDevices(bool silent=false)
{
	std::vector<gpu::Device> devices = gpu::enumDevices(true, true, silent);

	std::vector<gpu::Device> vk_devices;
	for (auto &device : devices) {
		if (device.supports_vulkan) {
			vk_devices.push_back(device);
		}
	}
	if (!silent) {
		std::cout << "Vulkan supported devices: " << vk_devices.size() << " out of " << devices.size() << std::endl;
	}

	rassert(vk_devices.size() > 0, 364773896);
	return vk_devices;
}

bool isEnabled(const std::string &env_variable_name, bool enabled_by_default)
{
	char *env_variable_value = getenv(env_variable_name.c_str()); // can be disabled with env variable <env_variable_name>=false
	if (env_variable_value) {
		if (env_variable_value == std::string("false")) {
			return false;
		} else if (env_variable_value == std::string("true")) {
			return true;
		} else {
			rassert(false, 556882154, "unrecognized env value (expected: true/false)", env_variable_name + "=" + env_variable_value);
		}
	} else {
		return enabled_by_default;
	}
}

bool isValidationLayersEnabled()
{
	// enable validation layers in Vulkan unit-tests, so we have them evaluated regularly on CI - preventing regressions
	return isEnabled("AVK_ENABLE_VALIDATION_LAYERS", true); // can be disabled with env variable AVK_ENABLE_VALIDATION_LAYERS=false
}

bool isMemoryGuardsEnabled()
{
	return isEnabled("AVK_ENABLE_MEMORY_GUARDS", true);
}

bool isMemoryGuardsChecksAfterKernelsEnabled()
{
	return isEnabled("AVK_ENABLE_MEMORY_GUARDS_CHECKS_AFTER_KERNELS", true);
}

gpu::Context activateVKContext(gpu::Device &device, bool silent=false)
{
	device.supports_opencl = false;
	device.supports_cuda = false;
	rassert(device.supports_vulkan, 771075327231479);

	if (!silent) {
		rassert(device.printInfo(), 7710753277608479);
	}

	gpu::Context gpu_context;
	gpu_context.initVulkan(device.device_id_vulkan);

	gpu_context.setVKValidationLayers(isValidationLayersEnabled());
	gpu_context.setMemoryGuards(isMemoryGuardsEnabled());
	gpu_context.setMemoryGuardsChecksAfterKernels(isMemoryGuardsChecksAfterKernelsEnabled());

	gpu_context.activate();
	return gpu_context;
}

void checkValidationLayerCallback()
{
	std::shared_ptr<avk2::InstanceContext> instance_context = avk2::InstanceContext::getGlobalInstanceContext(isValidationLayersEnabled());
	bool validation_errors_happend = instance_context->isDebugCallbackTriggered();
	if (validation_errors_happend) {
		instance_context->setDebugCallbackTriggered(false); // so that further tests will not fail because of previous test's validation errors
		rassert(!validation_errors_happend, "Validation layer detected a problem! 45124124321312");
	}
}

void checkNumberOfConstructedContexts()
{
	// let's ensure that one/two avk2::InstanceContext were constructed
	// the first one (without any validation layers) should be requested by VulkanEnum::enumDevices
	// and if validation layers are used - the second context also will be constructed
	std::shared_ptr<avk2::InstanceContext> instance_context = avk2::InstanceContext::getGlobalInstanceContext(isValidationLayersEnabled());
	rassert(instance_context->getConstructionIndex() == isValidationLayersEnabled(), 345124124123);
}

void checkPostInvariants()
{
	checkValidationLayerCallback();
	checkNumberOfConstructedContexts();
}

typedef std::mt19937 Random;

template <typename T>
T generate_random_color(Random &r, T min_value, T max_value)
{
	std::uniform_int_distribution<T> random_color(min_value, max_value);
	return random_color(r);
}

template <>
char generate_random_color<char>(Random &r, char min_value, char max_value)
{
	// this is to workaround MSVC compilation error (i.e. it can't compile std::uniform_int_distribution<char):
	// random(1863): error C2338: invalid template argument for uniform_int_distribution: N4659 29.6.1.1 [rand.req.genl]/1e requires one of short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long
	// error C2338: note: char, signed char, unsigned char, char8_t, int8_t, and uint8_t are not allowed
	std::uniform_int_distribution<int> random_color(min_value, max_value);
	int res_int = random_color(r);
	rassert(res_int >= min_value && res_int <= max_value, 117565785);
	return (char) res_int;
}

template <>
unsigned char generate_random_color<unsigned char>(Random &r, unsigned char min_value, unsigned char max_value)
{
	// this is to workaround MSVC compilation error (i.e. it can't compile std::uniform_int_distribution<unsigned char):
	// random(1863): error C2338: invalid template argument for uniform_int_distribution: N4659 29.6.1.1 [rand.req.genl]/1e requires one of short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long
	// error C2338: note: char, signed char, unsigned char, char8_t, int8_t, and uint8_t are not allowed
	std::uniform_int_distribution<int> random_color(min_value, max_value);
	int res_int = random_color(r);
	rassert(res_int >= min_value && res_int <= max_value, 117565785);
	return (unsigned char) res_int;
}

template <>
float generate_random_color<float>(Random &r, float min_value, float max_value)
{
	std::uniform_real_distribution<float> random_color(min_value, max_value);
	return random_color(r);
}

template <>
double generate_random_color<double>(Random &r, double min_value, double max_value)
{
	std::uniform_real_distribution<double> random_color(min_value, max_value);
	return random_color(r);
}

}