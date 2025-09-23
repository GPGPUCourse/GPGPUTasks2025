#pragma once

#define HOST_CODE 1
#include "common.vk"

#define VK_VERTICES_BINDING					0

#define VK_MAX_DESCRIPTORS_PER_TYPE			32 // increase if got Device::allocateDescriptorSets: ErrorOutOfPoolMemory
#define VK_POOL_DESCRIPTOR_TYPES			{vk::DescriptorType::eStorageBuffer, vk::DescriptorType::eStorageImage, vk::DescriptorType::eSampledImage, vk::DescriptorType::eCombinedImageSampler}

#define VK_GPU_MIN_VRAM_REQUIRED			(400*1024*1024) // i.e. we ignore devices with VRAM<400 MB, in fact the main motivation is to filter out integrated APU with 256 MB like described in ticket #5124

// see https://renderdoc.org/docs/in_application_api.html
// and https://github.com/baldurk/renderdoc/issues/1155
#define ENABLE_AND_ENSURE_RENDERDOC			0 // when enabled - application should be launched from RenderDoc GUI (because its shared library should be auto-preloaded in RenderDoc before application launch)
