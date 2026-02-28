#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> 
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(__global const uint* src,
                                           __global       uint* counts,
                                           unsigned int n,
                                           unsigned int shift)
{
    uint lid = get_local_id(0);
    uint gid = get_global_id(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);

    __local uint local_counts[16];
    if (lid < 16) {
        local_counts[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n) {
        uint bucket = (src[gid] >> shift) & 0xF;
        atomic_add(&local_counts[bucket], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 16) {
        counts[lid * num_groups + group_id] = local_counts[lid];
    }
}