#pragma once

#include <stdexcept>
#include "libbase/string_utils.h"

namespace gpu {

	class gpu_exception : public std::runtime_error {
	public:
		gpu_exception(std::string msg) throw ()					: runtime_error(msg)							{	}
		gpu_exception(const char *msg) throw ()					: runtime_error(msg)							{	}
		gpu_exception() throw ()								: runtime_error("GPU exception")				{	}
	};

	class gpu_bad_alloc : public gpu_exception {
	public:
		gpu_bad_alloc(std::string msg) throw ()					: gpu_exception(msg)							{	}
		gpu_bad_alloc(const char *msg) throw ()					: gpu_exception(msg)							{	}
		gpu_bad_alloc() throw ()								: gpu_exception("GPU exception")				{	}
	};

	void raiseException(std::string file, int line, std::string message);

	template <typename T>
	size_t deviceTypeSize();

	template <typename T>
	T deviceTypeMax();

	template <typename T>
	T deviceTypeMin();

	inline size_t divup(size_t num, size_t denom) {
		return (num + denom - 1) / denom;
	}

	unsigned int calcNChunk(size_t n, size_t group_size, size_t max_size=1000*1000);
	unsigned int calcColsChunk(size_t width, size_t height, size_t group_size_x, size_t max_size=1000*1000);
	unsigned int calcRowsChunk(size_t width, size_t height, size_t group_size_y, size_t max_size=1000*1000);
	unsigned int calcZSlicesChunk(size_t x, size_t y, size_t z, size_t group_size_z, size_t max_size=1000*1000);

	// see https://pcisig.com/membership/member-companies
	enum VENDOR {
		ID_AMD		= 0x1002,
		ID_INTEL	= 0x8086,
		ID_NVIDIA	= 0x10de,
		ID_APPLE	= 0x106b,
	};
}

#ifndef _SHORT_FILE_
#define _SHORT_FILE_ "unknown"
#endif

#define GPU_CHECKED_VERBOSE(x, message)	if (!(x)) {gpu::raiseException(_SHORT_FILE_, __LINE__, message);}
#define GPU_CHECKED(x)					if (!(x)) {gpu::raiseException(_SHORT_FILE_, __LINE__, "");}
