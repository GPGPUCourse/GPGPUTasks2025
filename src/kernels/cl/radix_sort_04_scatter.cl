#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global       uint* output,
    __global const uint* global_offsets,
    __global const uint* local_offsets,
    unsigned int bit,
    unsigned int n)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_sz = get_local_size(0);
    const unsigned int num_groups = get_num_groups(0) * get_num_groups(1);

    __local unsigned int ldata[RADIX];

    for (unsigned int i = local_id; i < RADIX; i += local_sz) {
        ldata[i] = global_offsets[i] + local_offsets[i * num_groups + group_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id >= n) return;

    unsigned int v = input[global_id];
    unsigned int b = (v >> bit) & (RADIX - 1u);

    unsigned int local_rank = 0;
    const unsigned int group_start = group_id * local_sz;
    for (unsigned int t = 0; t < local_id; ++t) {
        unsigned int b2 = (input[group_start + t] >> bit) & (RADIX - 1u);
        if (b2 == b) local_rank++;
    }
    output[ldata[b] + local_rank] = v;
}