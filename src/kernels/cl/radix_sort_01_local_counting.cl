#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* buffer,
    __global       uint* local_histograms,
    unsigned int n,
    unsigned int sorted_bit_offset)
{

    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_size = get_local_size(0);

    __local unsigned int local_counting[BUCKETS_COUNT];

    for (unsigned int j = local_id; j < BUCKETS_COUNT; j += group_size) {
        local_counting[j] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        unsigned int key = buffer[global_id];
        unsigned int bucket_idx = (key >> sorted_bit_offset) & (BUCKETS_COUNT - 1);
        atomic_inc(&local_counting[bucket_idx]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int base_index = group_id * BUCKETS_COUNT;
    for (unsigned int j = local_id; j < BUCKETS_COUNT; j += group_size) {
        local_histograms[base_index + j] = local_counting[j];
    }
}
