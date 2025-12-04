#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* restrict src,
    __global const uint* restrict group_offsets,
    __global uint* restrict dst,
    uint count,
    uint shift)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint bid = get_group_id(0);
    const uint num_bins = 1 << RADIX_BIT_CNT;
    const uint bin_mask = num_bins - 1;

    __local uint shared_bins[GROUP_SIZE * (1 << RADIX_BIT_CNT)];

    if (lid == 0) {
        uint val = src[bid * GROUP_SIZE];
        uint key = (val >> shift) & bin_mask;
        for (uint k = 0; k < num_bins; ++k) {
            shared_bins[k * GROUP_SIZE] = (key == k);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 1; i < GROUP_SIZE; ++i) {
        uint key = 0;
        uint src_idx = bid * GROUP_SIZE + i;
        if (src_idx < count) {
            key = (src[src_idx] >> shift) & bin_mask;
        }
        if (lid < num_bins) {
            uint idx = lid * GROUP_SIZE + i;
            shared_bins[idx] = shared_bins[idx - 1] + (key == lid);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < count) {
        uint element = src[gid];
        uint key = (element >> shift) & bin_mask;
        uint total_groups = (count + GROUP_SIZE - 1) / GROUP_SIZE;

        int block_ptr = (int)(key * total_groups + bid) - 1;
        uint global_offset = (block_ptr >= 0) ? group_offsets[block_ptr] : 0;

        int local_ptr = (int)(key * GROUP_SIZE + lid) - 1;
        uint local_offset = 0;
        if (lid > 0 && local_ptr >= 0) {
            local_offset = shared_bins[local_ptr];
        }

        dst[global_offset + local_offset] = element;
    }
}
