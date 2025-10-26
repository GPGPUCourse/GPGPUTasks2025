#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input_buf,
    __global const uint* reduc_buffer,
    __global       uint* output_buf,
    __global       uint* counts,
    const unsigned int n,
    const unsigned int shift)
{
    __local unsigned int reduc_data[NUM_BOXES];
    __local unsigned int cnt_data[NUM_BOXES];
    __local unsigned int local_data2[GROUP_SIZE];

    const unsigned int idx = get_global_id(0);
    const unsigned int local_idx = get_local_id(0);

    if (local_idx < NUM_BOXES) {
        reduc_data[local_idx] = reduc_buffer[(idx / GROUP_SIZE) * NUM_BOXES + local_idx];
        cnt_data[local_idx] = counts[local_idx];
    }
    
    const unsigned int val = (idx < n ? input_buf[idx] : 0u);
    const unsigned int mask_val = (val >> shift) & RADIX_MASK;
    local_data2[local_idx] = mask_val;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx < n) {
        unsigned int cnt = 0u;
        for (int i = local_idx; i < GROUP_SIZE && idx - local_idx + i < n; ++i) {
            if (local_data2[i] == mask_val) ++cnt;
        }

        unsigned int index = reduc_data[mask_val] - cnt;
        if (mask_val > 0u) index += cnt_data[mask_val - 1u];

        output_buf[index] = val;
    }
}
