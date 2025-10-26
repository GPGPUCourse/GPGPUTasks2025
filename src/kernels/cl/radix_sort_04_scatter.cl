#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefix_sum,
    __global uint* output,
    unsigned int n,
    unsigned int block_idx,
    unsigned int offset)
{
    unsigned int global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }
    unsigned int local_id = get_local_id(0);
    
    unsigned int idx = (input[global_id] >> offset) & (BLOCK_SIZE - 1);
    unsigned int prev_prefix = 0;
    unsigned int group_id = get_group_id(0);
    if (group_id > 0) {
        prev_prefix = prefix_sum[(group_id - 1) * BLOCK_SIZE + idx];
    }

    __local unsigned int local_prefix[GROUP_SIZE * BLOCK_SIZE];
    for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
        if (idx == i) {
            local_prefix[local_id * BLOCK_SIZE + idx] = 1;
        } else {
            local_prefix[local_id * BLOCK_SIZE + i] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int last_bucket = get_num_groups(0) - 1;
    __local unsigned int bucket_prefix[BLOCK_SIZE];

    if (local_id < BLOCK_SIZE) {
        for (unsigned int i = 1; i < GROUP_SIZE; i++) {
            local_prefix[i * BLOCK_SIZE + local_id] += local_prefix[(i - 1) * BLOCK_SIZE + local_id];
        }
        unsigned int prefix = 0;
        for (unsigned int b = 0; b < local_id; ++b) {
            prefix += prefix_sum[last_bucket * BLOCK_SIZE + b];
        }
        bucket_prefix[local_id] = prefix;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (global_id < n) {
        unsigned int pos = bucket_prefix[idx] + prev_prefix + local_prefix[local_id * BLOCK_SIZE + idx] - 1;
        output[pos] = input[global_id];   
    }
}

