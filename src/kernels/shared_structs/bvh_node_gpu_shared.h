#ifndef bvh_node_gpu_shared_pragma_once // pragma once
#define bvh_node_gpu_shared_pragma_once

#include "struct_helpers.h"

#include "aabb_gpu_shared.h"

/* Language-agnostic 32-bit unsigned */
#if defined(__OPENCL_VERSION__)
  /* OpenCL C */
  #define GPUC_UINT uint
#elif defined(common_vk)
  /* Vulkan GLSL */
  #define GPUC_UINT uint
#else
  /* C/C++/CUDA */
  #include <stdint.h>
  #define GPUC_UINT uint32_t
#endif

// POD AABB struct with identical layout in C++ / CUDA / OpenCL / Vulkan C-like code.
// Uses only scalar floats to avoid float3/vector alignment differences.
GPU_STRUCT_BEGIN(BVHNodeGPU)
    AABBGPU aabb;
    GPUC_UINT leftChildIndex;
    GPUC_UINT rightChildIndex;
    GPUC_UINT parentIndex;
GPU_STRUCT_END(BVHNodeGPU)

// Primitive face on gpu
GPU_STRUCT_BEGIN(PrimGPU)
    GPUC_UINT faceIndex;
    GPUC_UINT mortonCode;
GPU_STRUCT_END(PrimGPU)

/* ---------------- Host-only layout checks ---------------- */
#if !defined(__OPENCL_VERSION__)
  /* These static_asserts are ignored in OpenCL C.
     They guarantee identical, padding-free layout for host/CUDA. */
  #if defined(__cplusplus)
    static_assert(sizeof(GPUC_UINT) == 4, "GPUC_UINT must be 32-bit");

    static_assert(sizeof(BVHNodeGPU) == sizeof(AABBGPU) + 3*4, "BVHNodeGPU size mismatch");
  #endif
#endif

#endif // pragma once
