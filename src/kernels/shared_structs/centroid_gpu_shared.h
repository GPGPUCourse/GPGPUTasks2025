#ifndef centroids_gpu_shared_pragma_once // pragma once
#define centroids_gpu_shared_pragma_once

#include "struct_helpers.h"

GPU_STRUCT_BEGIN(CentroidGPU)
float x;
float y;
float z;
GPU_STRUCT_END(CentroidGPU)

/* ---------------- Host-only layout checks ---------------- */
#if !defined(__OPENCL_VERSION__) && !defined(common_vk)
/* These static_asserts are ignored in OpenCL C.
   They guarantee identical, padding-free layout for host/CUDA. */
#if defined(__cplusplus)
static_assert(sizeof(float) == 4, "float must be 32-bit");

static_assert(sizeof(CentroidGPU) == 3 * 4, "CentroidsGPU size mismatch");
#endif
#endif

#endif // pragma once
