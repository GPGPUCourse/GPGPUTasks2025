#pragma once

#include <libgpu/utils.h>

namespace avk2 {

	class vk_exception : public gpu::gpu_exception {
	public:
		vk_exception(std::string msg) throw ()			: gpu_exception(msg)							{	}
		vk_exception(const char *msg) throw ()			: gpu_exception(msg)							{	}
		vk_exception() throw ()							: gpu_exception("Vulkan exception")				{	}
	};

	class vk_bad_alloc : public gpu::gpu_bad_alloc {
	public:
		vk_bad_alloc(std::string msg) throw ()			: gpu_bad_alloc(msg)							{	}
		vk_bad_alloc(const char *msg) throw ()			: gpu_bad_alloc(msg)							{	}
		vk_bad_alloc() throw ()							: gpu_bad_alloc("Vulkan exception")				{	}
	};

}
