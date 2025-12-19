#ifndef bvh_node_gpu_shared_pragma_once // pragma once
#define bvh_node_gpu_shared_pragma_once

#include "struct_helpers.h"

#include "aabb_gpu_shared.h"

/* Language-agnostic 32-bit unsigned */
#if defined(__OPENCL_VERSION__)
  /* OpenCL C */
  #define GPUC_UINT uint
  #define GPUC_INT int
#elif defined(common_vk)
  /* Vulkan GLSL */
  #define GPUC_UINT uint
  #define GPUC_INT int
#else
  /* C/C++/CUDA */
  #include <stdint.h>
  #define GPUC_UINT uint32_t
  #define GPUC_INT int32_t
#endif

// POD AABB struct with identical layout in C++ / CUDA / OpenCL / Vulkan C-like code.
// Uses only scalar floats to avoid float3/vector alignment differences.
GPU_STRUCT_BEGIN(BVHNodeGPU)
    AABBGPU aabb;
    GPUC_UINT leftChildIndex;
    GPUC_UINT rightChildIndex;
GPU_STRUCT_END(BVHNodeGPU)

GPU_STRUCT_BEGIN(BVHThinNodeGPU)
    AABBGPU aabb;
    GPUC_INT children[2];
GPU_STRUCT_END(BVHThinNodeGPU)

// почему-то в области трассировки лучей документация RDNA3 написана совсем плохо
// упоминается, что BVH может содержать узлы видов Box (4 AABB) и Triangle (один треугольник)
// однако эти структуры совершенно не задокументированы
// к счастью есть исходники драйверов. взято из mesa из src/amd/vulkan/bvh/bvh.h
GPU_STRUCT_BEGIN(BVHTriangleGPU)
    float a[3], b[3], c[3];
    GPUC_UINT reserved[3];
    GPUC_UINT triangle_id;
    GPUC_UINT geometry_id_and_flags;
    GPUC_UINT reserved2;
    GPUC_UINT id;
GPU_STRUCT_END(BVHTriangleGPU)

GPU_STRUCT_BEGIN(BVHBoxGPU)
    GPUC_UINT children[4];
    AABBGPU coords[4];
    GPUC_UINT flags;
    GPUC_UINT reserved[3];
GPU_STRUCT_END(BVHBoxGPU)
// отличаются эти структуры по нижним трём битам указателей на них (конечно, в документации этого тоже нету)
#define BVH_TRIANGLE_PTR_TAG 0
#define BVH_BOX_PTR_TAG 5 // есть ещё на float16 с тегом 4, но не будем таким заниматься
// также размеры всех структур кратны 64, а указатели сдвинуты вправо на 3 чтобы не тратить лишние биты
// ещё в документации не упомянуты узлы AABB (???) и Instance (хранит матрицу преобразования, обратную ей и указатель на другой BVH)
// аппаратно они обрабатываются или программно в коде генерируемом для OpTraceRayKHR я понять не смог (возможно, и то и другое), но мне они не нужны
// мы кладём немного своего мусора (указатели на родителей) в triangle.id и box.reserved[2], вроде не ломается

/* ---------------- Host-only layout checks ---------------- */
#if !defined(__OPENCL_VERSION__)
  /* These static_asserts are ignored in OpenCL C.
     They guarantee identical, padding-free layout for host/CUDA. */
  #if defined(__cplusplus)
    static_assert(sizeof(GPUC_UINT) == 4, "GPUC_UINT must be 32-bit");

    static_assert(sizeof(BVHNodeGPU) == sizeof(AABBGPU) + 2*4, "BVHNodeGPU size mismatch");
    static_assert(sizeof(BVHTriangleGPU) == 64);
    static_assert(sizeof(BVHBoxGPU) == 128);
  #endif
#endif

#endif // pragma once
