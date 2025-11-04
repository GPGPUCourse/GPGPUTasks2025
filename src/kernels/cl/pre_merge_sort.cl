#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
pre_merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int n)
{
    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);
    __local uint buffer[GROUP_SIZE];
    if (global_id < n) {
        buffer[local_id] = input_data[global_id];
    } else {
        buffer[local_id] = INF;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for (uint block_sz = 1; block_sz < GROUP_SIZE; block_sz *= 2) {
        const uint block_num = local_id / block_sz;
        uint l = 0;
        uint r = block_sz + 1;
        const bool include_equal = block_num & 1;
        const uint search_block_start = (block_num ^ 1) * block_sz;
        const uint my_number = buffer[local_id];
        while (l + 1 != r) {
            const uint m = (l + r) >> 1;
            const uint val = buffer[search_block_start + m - 1];
            if (val < my_number || (include_equal && val == my_number)) {
                l = m;
            } else {
                r = m;
            }
        }
        const uint res_index = l + (local_id - block_num * block_sz);
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint workspace_start = (block_num - (block_num & 1)) * block_sz;
        buffer[workspace_start + res_index] = my_number;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_id < n) {
        output_data[global_id] = buffer[local_id];
    }
}
