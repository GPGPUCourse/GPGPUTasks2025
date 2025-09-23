#pragma once

#include <libgpu/utils.h>

namespace ocl {

	class ocl_exception : public gpu::gpu_exception {
	public:
		ocl_exception(std::string msg) throw ()					: gpu_exception(msg)							{	}
		ocl_exception(const char *msg) throw ()					: gpu_exception(msg)							{	}
		ocl_exception() throw ()								: gpu_exception("OpenCL exception")				{	}
	};

	class ocl_bad_alloc : public gpu::gpu_bad_alloc {
	public:
		ocl_bad_alloc(std::string msg) throw ()					: gpu_bad_alloc(msg)							{	}
		ocl_bad_alloc(const char *msg) throw ()					: gpu_bad_alloc(msg)							{	}
		ocl_bad_alloc() throw ()								: gpu_bad_alloc("OpenCL exception")				{	}
	};

}
