#ifndef denoise_gpu_shared_pragma_once // pragma once
#define denoise_gpu_shared_pragma_once

#include "struct_helpers.h"

// POD AABB struct with identical layout in C++ / CUDA / OpenCL / Vulkan C-like code.
// Uses only scalar floats to avoid float3/vector alignment differences.
GPU_STRUCT_BEGIN(DENOISE)
float value_sum;
int samples;
GPU_STRUCT_END(DENOISE)

/* ---------------- Host-only layout checks ---------------- */
#if !defined(__OPENCL_VERSION__) && !defined(common_vk)
/* These static_asserts are ignored in OpenCL C.
   They guarantee identical, padding-free layout for host/CUDA. */
#if defined(__cplusplus)
static_assert(sizeof(float) == 4, "float must be 32-bit");
static_assert(sizeof(int) == 4, "int must be 32-bit");

static_assert(sizeof(DENOISE) == 2 * 4, "DENOISE size mismatch");
#endif
#endif

#endif // pragma once
