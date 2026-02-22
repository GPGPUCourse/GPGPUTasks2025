#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1              
){
    const size_t n = a1;

    const size_t local_id = get_local_id(0);
    const size_t group_id = get_group_id(0);

    const size_t elems_per_block = 2 * GROUP_SIZE;
    const size_t base = group_id * elems_per_block;

    const size_t idx0 = base + local_id;
    const size_t idx1 = base + GROUP_SIZE + local_id;

    uint v0 = 0;
    uint v1 = 0;
    if (idx0 < n) {
        v0 = buffer1[idx0];
    }
    if (idx1 < n) {
        v1 = buffer1[idx1];
    }

    __local uint s[2 * GROUP_SIZE];
    s[local_id] = v0;
    s[GROUP_SIZE + local_id] = v1;
    barrier(CLK_LOCAL_MEM_FENCE);

    // upsweep
    for (size_t stride = 1; stride < elems_per_block; stride <<= 1) {
        size_t index = ((local_id + 1) * stride * 2) - 1;
        if (index < elems_per_block) {
            s[index] += s[index - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        buffer2[group_id] = s[elems_per_block - 1];
        s[elems_per_block - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // downsweep
    for (size_t stride = elems_per_block >> 1; stride >= 1; stride >>= 1) {
        size_t index = ((local_id + 1) * stride * 2) - 1;
        if (index < elems_per_block) {
            uint t = s[index - stride];
            s[index - stride] = s[index];
            s[index] += t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}