#ifndef aabb_gpu_shared_pragma_once // pragma once
#define aabb_gpu_shared_pragma_once

#include "struct_helpers.h"

// POD AABB struct with identical layout in C++ / CUDA / OpenCL / Vulkan C-like code.
// Uses only scalar floats to avoid float3/vector alignment differences.
GPU_STRUCT_BEGIN(AABBGPU)
// Minimum corner of the box
float min_x;
float min_y;
float min_z;

// Maximum corner of the box
float max_x;
float max_y;
float max_z;
GPU_STRUCT_END(AABBGPU)

/* ---------------- Host-only layout checks ---------------- */
#if !defined(__OPENCL_VERSION__) && !defined(common_vk)
/* These static_asserts are ignored in OpenCL C.
   They guarantee identical, padding-free layout for host/CUDA. */
#if defined(__cplusplus)
static_assert(sizeof(float) == 4, "float must be 32-bit");

static_assert(sizeof(AABBGPU) == 6 * 4, "AABBGPU size mismatch");
#endif
#endif

#endif // pragma once
