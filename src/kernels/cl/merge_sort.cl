#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    uint block_sz,
    uint n)
{
    const uint global_id = get_global_id(0);
    if (global_id < n) {
        const uint block_num = global_id / block_sz;
        const uint my_idx_in_block = global_id % block_sz;
        const bool include_equal = block_num & 1;
        const uint search_block_start = (block_num ^ 1) * block_sz;
        const uint my_number = input_data[global_id];
        if (search_block_start >= n) {
            output_data[global_id] = my_number;
            return;
        }
        const uint other_block_size = min(search_block_start + block_sz, n) - search_block_start;
        uint l = 0;
        uint r = other_block_size + 1;

        while (l + 1 < r) {
            const uint m = (l + r) >> 1;
            const uint val = input_data[search_block_start + m - 1];
            if (val < my_number || (include_equal && val == my_number)) {
                l = m;
            } else {
                r = m;
            }
        }
        const uint res_index = l + my_idx_in_block;
        const uint workspace_start = (block_num - (block_num & 1)) * block_sz;
        output_data[workspace_start + res_index] = my_number;
    }
}
