#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global       uint* local_hist,
    unsigned int bit,
    unsigned int n
) {
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int global_sz = get_global_size(0);
    const unsigned int local_sz = get_local_size(0);

    __local unsigned int data[RADIX];
    for (unsigned int i = local_id; i < RADIX; i += local_sz) data[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = global_id; i < n; i += global_sz) {
        unsigned int b = (input[i] >> bit) & (RADIX - 1u);
        atomic_add((volatile __local unsigned int*)&data[b], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = local_id; i < RADIX; i += local_sz) {
        local_hist[i * get_num_groups(0) * get_num_groups(1) + group_id] = data[i];
    }
}