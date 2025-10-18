#ifndef opencl_translator_cu // pragma once
#define opencl_translator_cu

#ifdef __NVCC__

#ifndef STATIC_KEYWORD
#define STATIC_KEYWORD __device__
#endif

// See https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/functionQualifiers.html
#define vec_type_hint(typen)
#define work_group_size_hint(X, Y, Z)
#define reqd_work_group_size(X, Y, Z)

#define __kernel __global__
#define __global
#define __local __shared__
#define __constant __constant__

typedef unsigned int uint;

// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/barrier.html
enum	cl_mem_fence_flags
{
    CLK_LOCAL_MEM_FENCE,
    CLK_GLOBAL_MEM_FENCE
};

STATIC_KEYWORD void	barrier(cl_mem_fence_flags flags)
{
    __syncthreads();
}

// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/workItemFunctions.html
STATIC_KEYWORD size_t	getXYZByIndex(dim3 xyz, uint dimindx)
{
    if (dimindx == 2) {
        return xyz.z;
    } else if (dimindx == 1) {
        return xyz.y;
    } else {
        return xyz.x;
    }
}

STATIC_KEYWORD size_t	get_global_size	(uint dimindx) {
    return getXYZByIndex(gridDim, dimindx) * getXYZByIndex(blockDim, dimindx);
}

STATIC_KEYWORD size_t	get_global_id	(uint dimindx) {
    return getXYZByIndex(blockIdx, dimindx) * getXYZByIndex(blockDim, dimindx) + getXYZByIndex(threadIdx, dimindx);
}

STATIC_KEYWORD size_t	get_local_size	(uint dimindx) {
    return getXYZByIndex(blockDim, dimindx);
}

STATIC_KEYWORD size_t	get_local_id	(uint dimindx) {
    return getXYZByIndex(threadIdx, dimindx);
}

STATIC_KEYWORD size_t	get_num_groups	(uint dimindx) {
    return getXYZByIndex(gridDim, dimindx);
}

STATIC_KEYWORD size_t	get_group_id	(uint dimindx) {
    return getXYZByIndex(blockIdx, dimindx);
}

STATIC_KEYWORD uint	get_work_dim() 
{
    if (get_global_size(2) > 1) {
        return 3;
    } else if (get_global_size(1) > 1) {
        return 2;
    } else {
        return 1;
    }
}

STATIC_KEYWORD int atomic_add(int *p, int val) {
    return atomicAdd(p, val);
}

STATIC_KEYWORD float atomic_add(float *p, float val) {
    return atomicAdd(p, val);
}

#define WARP_SIZE 32

#endif

#ifdef __CUDA_ARCH__
#define DEVICE_CODE
#else
#define HOST_CODE
#endif

#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/utils.h>
#include <cuda_runtime_api.h>

#endif // pragma once
