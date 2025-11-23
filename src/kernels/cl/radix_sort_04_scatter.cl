#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* local_prefix_sums,
    __global       uint* output,
    unsigned int byte_index,
    unsigned int n)
{
    const unsigned int global_id = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    
    __local uint local_prefix[RADIX_BUCKET_COUNT];
    
    if (local_id < RADIX_BUCKET_COUNT) {
        local_prefix[local_id] = local_prefix_sums[group_id * RADIX_BUCKET_COUNT + local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        uint value = values[global_id];
        uint byte_value = (value >> (byte_index * 8)) & 0xFF;
        uint position = atomic_inc(&local_prefix[byte_value]);
        output[position] = value;
    }
}
