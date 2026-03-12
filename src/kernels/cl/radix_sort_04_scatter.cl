#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
        __global const uint* input,
        __global const uint* prefix_sum_accum,
        __global uint* output,
        unsigned int n,
        unsigned int offset)
{
    const uint index = get_global_id(0);
    const uint group = get_group_id(0);
    const uint local_id = get_local_id(0);
    const uint group_size = get_local_size(0);

    if (index >= n)
        return;

    __local uint local_prefix[GROUP_SIZE];

    const uint inp_bit = input[index];
    const bool last_bit = ((input[index] >> offset) & 1) == 1;

    uint is_zero = last_bit ? 0 : 1;

    local_prefix[local_id] = is_zero;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 1; stride < group_size; stride <<= 1) {
        uint val = 0;
        if (local_id >= stride) {
            val = local_prefix[local_id - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        local_prefix[local_id] += val;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint local_offset, global_offset;

    if (last_bit) {
        local_offset = (local_id + 1) - local_prefix[local_id];
        global_offset = prefix_sum_accum[n - 1];

        if (group > 0) {
            global_offset += (group * group_size) - prefix_sum_accum[group * group_size - 1];
        }

        output[global_offset + local_offset - 1] = inp_bit;
    } else {
        local_offset = local_prefix[local_id];

        if (group > 0) {
            global_offset = prefix_sum_accum[group * group_size - 1];
        } else {
            global_offset = 0;
        }

        output[global_offset + local_offset - 1] = inp_bit;
    }
}