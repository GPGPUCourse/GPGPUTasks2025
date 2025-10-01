#pragma once

// This header is very large - DON'T include it from headers, use forward declarations instead

// MoltenVK 1.3.283 on MacOS provides support only for Vulkan 1.2, so we need to support Vulkan 1.2 
#define VULKAN_MIN_VERSION_MAJOR 1
#define VULKAN_MIN_VERSION_MINOR 2
#define VULKAN_MIN_VERSION_PATCH 0
#define VULKAN_MIN_VERSION VK_MAKE_API_VERSION(0, VULKAN_MIN_VERSION_MAJOR, VULKAN_MIN_VERSION_MINOR, VULKAN_MIN_VERSION_PATCH)

//______________________________________________________________________________________________________________________
// Vulkan API C++ bindings

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VK_NO_PROTOTYPES // disabled because prototypes are useful only for static-linking, but we are using dynamic-linking
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_beta.h> // we need to use VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME on MacOS+MoltenVK

#define VKF VULKAN_HPP_DEFAULT_DISPATCHER

namespace vk {
	// this is to workaround MSVC compilation error (which happens due to ambiguous call to vk::ArrayWrapper1D::operator std::string() VS operator T=char const *()):
	// error C2440: '<function-style-cast>': cannot convert from 'vk::ArrayWrapper1D<char,256>' to 'std::string'
	// note: No constructor could take the source type, or constructor overload resolution was ambiguous
	template <size_t N>
	std::string to_string(const vk::ArrayWrapper1D<char, N> &a) {
		return a.operator std::string();
	}
}

//______________________________________________________________________________________________________________________
// VMA - Vulkan Memory Allocator - https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/

#include "utils.h"
#define VMA_ASSERT(condition) avk2::reportVmaAssert(condition, __LINE__)

#define VMA_VULKAN_VERSION (VULKAN_MIN_VERSION_MAJOR*1000000+VULKAN_MIN_VERSION_MINOR*1000)
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vma/vk_mem_alloc.h>

//______________________________________________________________________________________________________________________

#define VK_LAYER_KHRONOS_VALIDATION_NAME "VK_LAYER_KHRONOS_validation"
