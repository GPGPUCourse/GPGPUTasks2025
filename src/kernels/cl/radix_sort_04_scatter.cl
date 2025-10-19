#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input_arr,
    __global const uint* bucket_pref_sums,
    __global const uint* bucket_sums,
    __global       uint* output_arr,
    unsigned int start_bit,
    unsigned int n,
    unsigned int blocks_cnt)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    __local uint bucket_offsets[RADIX_BUCKETS_CNT];
    if (lid < RADIX_BUCKETS_CNT) {
        if (lid > 0) {
            bucket_offsets[lid] = bucket_sums[lid - 1];
        } else {
            bucket_offsets[lid] = 0;
        }
    }

    __local uint group_bucket_offsets[RADIX_BUCKETS_CNT];
    if (lid < RADIX_BUCKETS_CNT) {
        uint group_id = get_group_id(0);
        if (group_id > 0) {
            group_bucket_offsets[lid] = bucket_pref_sums[lid * blocks_cnt + group_id - 1];
        } else {
            group_bucket_offsets[lid] = 0;
        }
    }

    uint elem = 0;
    uint elem_bucket = 0;
    if (gid < n) {
        elem = input_arr[gid];
        elem_bucket = elem >> start_bit & (RADIX_BUCKETS_CNT - 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    uint group_bucket_index;
    __local uint bucket_pref_sum[GROUP_SIZE];  // temporary
    for (uint bucket = 0; bucket < RADIX_BUCKETS_CNT; ++bucket) {
        bucket_pref_sum[lid] = (elem_bucket == bucket) ? 1 : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint offset = 1; offset < GROUP_SIZE; offset <<= 1) {
            uint temp = 0;
            if (lid >= offset)
                temp = bucket_pref_sum[lid - offset];
            barrier(CLK_LOCAL_MEM_FENCE);
            bucket_pref_sum[lid] += temp;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (elem_bucket == bucket) {
            group_bucket_index = bucket_pref_sum[lid] - 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < n) {
        uint bucket = elem_bucket;
        uint output_index = bucket_offsets[bucket] + group_bucket_offsets[bucket] + group_bucket_index;
        output_arr[output_index] = elem;
    }

    // if (lid == 0) {
    //     uint local_bucket_counters[RADIX_BUCKETS_CNT];
    //     for (int i = 0; i < RADIX_BUCKETS_CNT; ++i) {
    //         local_bucket_counters[i] = 0;
    //     }
    //     for (uint i = 0; i < GROUP_SIZE && gid + i < n; ++i) {
    //         uint value = input_arr[gid + i];
    //         uint bucket_index = (value >> start_bit) & (RADIX_BUCKETS_CNT - 1);
    //         uint output_index = bucket_offsets[bucket_index] + group_bucket_offsets[bucket_index] + local_bucket_counters[bucket_index];
    //         output_arr[output_index] = value;
    //         local_bucket_counters[bucket_index] += 1;
    //     }
    // }
}