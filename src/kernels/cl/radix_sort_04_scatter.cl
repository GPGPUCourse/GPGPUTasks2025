#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* array,
    __global const uint* prefix_sums,
    __global uint* result,
    unsigned int n,
    unsigned int bit_start)
{
    const unsigned int index = get_global_id(0);
    const unsigned int local_idx = get_local_id(0);
    const unsigned int group = get_group_id(0);
    __local unsigned int local_pref[GROUP_SIZE * (1 << RADIX_BIT_CNT)];
    __local unsigned int bucket_pref[1 << RADIX_BIT_CNT];
    if (local_idx == 0) {
        int bucket = (array[group * GROUP_SIZE] >> bit_start) & ((1 << RADIX_BIT_CNT) - 1);
        for (int i = 0; i < 1 << RADIX_BIT_CNT; i++) {
            local_pref[i * GROUP_SIZE] = (bucket == i ? 1 : 0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 1; i < GROUP_SIZE; i++) {
        int bucket = 0;
        if (i + group * GROUP_SIZE < n) {
            bucket = (array[i + group * GROUP_SIZE] >> bit_start) & ((1 << RADIX_BIT_CNT) - 1);
        }
        if (local_idx < 1 << RADIX_BIT_CNT) {
            local_pref[local_idx * GROUP_SIZE + i] = local_pref[local_idx * GROUP_SIZE + i - 1] + (bucket == local_idx ? 1 : 0);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (index < n) {
        int bucket = (array[index] >> bit_start) & ((1 << RADIX_BIT_CNT) - 1);
        int tmp = bucket * ((n + GROUP_SIZE - 1) / GROUP_SIZE) + group - 1;
        unsigned int global_pref = (tmp >= 0 ? prefix_sums[tmp] : 0);
        tmp = bucket * GROUP_SIZE + local_idx - 1;
        unsigned int local_pref_add = (local_idx > 0 && tmp >= 0 ? local_pref[tmp] : 0);
        unsigned int local_min = (bucket >= 0 ? local_pref[bucket * GROUP_SIZE - 1] : 0);
        unsigned int next_id = global_pref + local_pref_add;
        result[next_id] = array[index];
    }
}