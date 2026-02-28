#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> 
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(__global const uint* src,
                                    __global const uint* counts_scan,
                                    __global       uint* dst,
                                    unsigned int n,
                                    unsigned int shift)
{
    uint lid = get_local_id(0);
    uint gid = get_global_id(0);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);
    __local uint local_val[GROUP_SIZE];
    local_val[lid] = (gid < n) ? src[gid] : 0;
    __local uint global_offsets[16];
    if (lid < 16) {
        uint flat_id = lid * num_groups + group_id;
        global_offsets[lid] = (flat_id == 0) ? 0 : counts_scan[flat_id - 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n) {
        uint val = local_val[lid];
        uint bucket = (val >> shift) & 0xF;
        uint local_offset = 0;
        for (uint i = 0; i < lid; ++i) {
            if (((local_val[i] >> shift) & 0xF) == bucket) {
                local_offset++;
            }
        }
        dst[global_offsets[bucket] + local_offset] = val;
    }
}