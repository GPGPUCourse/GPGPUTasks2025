#pragma once

#include <libgpu/utils.h>

namespace cuda {

	class cuda_exception : public gpu::gpu_exception {
	public:
		cuda_exception(std::string msg) throw ()					: gpu_exception(msg)							{	}
		cuda_exception(const char *msg) throw ()					: gpu_exception(msg)							{	}
		cuda_exception() throw ()									: gpu_exception("CUDA exception")				{	}
	};

	class cuda_bad_alloc : public gpu::gpu_bad_alloc {
	public:
		cuda_bad_alloc(std::string msg) throw ()					: gpu_bad_alloc(msg)							{	}
		cuda_bad_alloc(const char *msg) throw ()					: gpu_bad_alloc(msg)							{	}
		cuda_bad_alloc() throw ()									: gpu_bad_alloc("CUDA exception")				{	}
	};

}