#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* buffer,
    __global const uint* histograms_pref_sum,
    __global       uint* scattered_buffer,
    unsigned int n,
    unsigned int sorted_bit_offset)
{
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int num_groups = get_num_groups(0);
    unsigned int group_size = get_local_size(0);

    __local unsigned int local_counts[BUCKETS_COUNT];
    __local unsigned int global_bucket_offsets[BUCKETS_COUNT]; // вот тут сделать префиксную сумму по последнему блоку histograms_pref_sum 

    for (unsigned int j = local_id; j < BUCKETS_COUNT; j += group_size) {
        local_counts[j] = 0u;
        global_bucket_offsets[j] = 0u;
    }
    barrier(CLK_LOCAL_MEM_FENCE);



    if (local_id == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < BUCKETS_COUNT; ++i) {
            global_bucket_offsets[i] = sum;
            sum += histograms_pref_sum[(num_groups - 1) * BUCKETS_COUNT + i];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_id < n) {
        unsigned int key = buffer[global_id];
        unsigned int bucket_idx = (key >> sorted_bit_offset) & (BUCKETS_COUNT - 1);
        
        unsigned int offset_from_prev_groups = (group_id == 0) ? 0 : histograms_pref_sum[(group_id - 1) * BUCKETS_COUNT + bucket_idx];

        unsigned int local_offset = atomic_add(&local_counts[bucket_idx], 1u);

        unsigned int global_index = offset_from_prev_groups + local_offset + global_bucket_offsets[bucket_idx];

        scattered_buffer[global_index] = key;
    }
}