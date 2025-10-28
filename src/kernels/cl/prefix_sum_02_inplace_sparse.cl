#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(BUILD_GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_inplace_sparse(
    __global const uint* in,
    __global uint* out,
    unsigned int n,
    unsigned int level)
{
    __local uint data[BUILD_GROUP_SIZE];
    size_t i = get_global_id(0);
    size_t local_idx = get_local_id(0);
    size_t initial_idx = (i + 1) * level - 1;
    if (initial_idx < n) {
        data[local_idx] = in[initial_idx];
    } else {
        data[local_idx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int iter = 1; iter <= BUILD_GROUP_SIZE_LOG; ++iter) {
        if (((local_idx + 1) & ((1 << iter) - 1)) == 0) {
            data[local_idx] += data[local_idx - (1 << (iter - 1))];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (initial_idx < n) {
        out[initial_idx] = data[local_idx];
    }
}
