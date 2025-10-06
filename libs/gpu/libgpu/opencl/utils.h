#pragma once

#define CL_TARGET_OPENCL_VERSION 210
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "exceptions.h"
#include "libbase/string_utils.h"
#include <libgpu/device.h>
#include <libgpu/utils.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdexcept>

namespace ocl {

#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE	0x11B3  // since OpenCL 1.1

#define CL_NV_DEVICE_ATTRIBUTE_QUERY_EXT				"cl_nv_device_attribute_query"
#define CL_AMD_DEVICE_ATTRIBUTE_QUERY_EXT				"cl_amd_device_attribute_query"

#ifndef CL_DEVICE_PCI_BUS_ID_NV
#define CL_DEVICE_PCI_BUS_ID_NV							0x4008
#endif

#ifndef CL_DEVICE_PCI_SLOT_ID_NV
#define CL_DEVICE_PCI_SLOT_ID_NV						0x4009
#endif

typedef union
{
	struct {
		cl_uint type;
		cl_uint data[5];
	} raw;
	struct {
		cl_uint type;
		cl_char unused[17];
		cl_char bus;
		cl_char device;
		cl_char function;
	} pcie;
} cl_device_topology_amd;

#ifndef CL_DEVICE_TOPOLOGY_AMD
#define CL_DEVICE_TOPOLOGY_AMD							0x4037
#endif

#ifndef CL_DEVICE_BOARD_NAME_AMD
#define CL_DEVICE_BOARD_NAME_AMD						0x4038
#endif

#ifndef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
#define CL_DEVICE_GLOBAL_FREE_MEMORY_AMD				0x4039
#endif

#ifndef CL_DEVICE_WAVEFRONT_WIDTH_AMD
#define CL_DEVICE_WAVEFRONT_WIDTH_AMD					0x4043
#endif

std::string errorString(cl_int code);

static inline void reportError(cl_int err, int line, const std::string &prefix = std::string())
{
	if (CL_SUCCESS == err)
		return;

	std::string message = prefix + errorString(err) + " (" + to_string(err) + ")" + " at line " + to_string(line);

	switch (err) {
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		throw ocl_bad_alloc(message);
	default:
		throw ocl_exception(message);
	}
}

#define OCL_SAFE_CALL(expr) ocl::reportError(expr, __LINE__, "")
#define OCL_SAFE_CALL_MESSAGE(expr, message) ocl::reportError(expr, __LINE__, message)
#define OCL_TRACE(expr) expr;

#define OCL_NOTHROW(expr) \
{ \
	try { \
		expr; \
	} \
	catch (std::exception &e) { \
		std::cerr << "Error: " << e.what() << std::endl; \
	} \
}

}
