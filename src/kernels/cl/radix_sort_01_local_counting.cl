#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* array,
    __global       uint* buckets,
    unsigned int n,
    unsigned int bit_start)
{
    const unsigned int index = get_global_id(0);
    const unsigned int local_idx = get_local_id(0);
    const unsigned int group = get_group_id(0);
    __local unsigned int local_buckets[1 << RADIX_BIT_CNT];
    if (local_idx < (1 << RADIX_BIT_CNT)) {
        local_buckets[local_idx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (index < n) {
        int bucket = (array[index] >> bit_start) & ((1 << RADIX_BIT_CNT) - 1);
        atomic_inc(&local_buckets[bucket]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_idx == 0) {
        for (unsigned int bucket = 0; bucket < 1 << RADIX_BIT_CNT; bucket++) {
            buckets[((n + GROUP_SIZE - 1) / GROUP_SIZE) * bucket + group] = local_buckets[bucket];
        }
    }
}
