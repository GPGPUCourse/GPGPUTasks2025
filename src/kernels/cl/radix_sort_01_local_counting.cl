#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    __global const uint* input,
    __global uint* bins,
    __global uint* scan,
    __global uint* scatter,
    const uint offset,
    const uint compressed,
    const uint n)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint local_start = index - local_index;
    const int remaining = local_start + GROUP_SIZE - n;

    __local uint local_input[GROUP_SIZE];
    __local uint local_accum[GROUP_SIZE];
    __local uint local_scan[GROUP_SIZE];

    const uint cur_input = (index < n ? input[index] : MAX);
    local_input[local_index] = cur_input;
    const uint cur_bin = (cur_input >> (offset * RADIX)) & RADIX_MASK;
    bins[index] = cur_bin;

#pragma unroll
    for (uint bin = 0; bin < BIN_COUNT; bin++) {
        local_accum[local_index] = (bin == cur_bin);
        barrier(CLK_LOCAL_MEM_FENCE);
        const uint index_offset = bin * GROUP_SIZE;
#pragma unroll
        for (uint k = 0; k <= LOG_GROUP_SIZE; k++) {
            const uint divided = (local_index + 1) >> k;
            const uint is_adding = (divided & 1);
            local_scan[local_index] = (k > 0 ? local_scan[local_index] : 0) + (is_adding ? local_accum[divided - 1] : 0);

            const uint next = local_index << 1;
            const uint to_write = next < (GROUP_SIZE >> k) ? (local_accum[next] + local_accum[next + 1]) : 0;
            barrier(CLK_LOCAL_MEM_FENCE);

            local_accum[local_index] = to_write;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_index == 0) {
            scan[compressed * bin + (local_start / GROUP_SIZE)] = local_scan[GROUP_SIZE - 1] - ((remaining > 0 && bin == BIN_COUNT - 1) ? remaining : 0);
        }
        if (bin == cur_bin) {
            scatter[index] = local_scan[local_index] - 1;
        }
    }
}
