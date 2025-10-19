#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* arr,
    __global       uint* bucket_counters,
    unsigned int blocks_cnt,
    unsigned int start_bit,
    unsigned int n)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    __local uint buckets[RADIX_BUCKETS_CNT];
    if (lid < RADIX_BUCKETS_CNT) {
        buckets[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (gid < n) {
        uint value = arr[gid];
        uint bucket_index = (value >> start_bit) & (RADIX_BUCKETS_CNT - 1);
        atomic_inc(&buckets[bucket_index]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < RADIX_BUCKETS_CNT) {
        bucket_counters[lid * blocks_cnt + get_group_id(0)] = buckets[lid];
    }
}
