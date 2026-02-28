#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(__global const uint* in_arr,
                                          __global       uint* out_arr, 
                                          __global       uint* block_sums, 
                                          unsigned int n) 
{
    uint lid = get_local_id(0);
    uint gid = get_global_id(0);
    uint group_id = get_group_id(0);

    __local uint temp[GROUP_SIZE];
    temp[lid] = (gid < n) ? in_arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = 1; offset < GROUP_SIZE; offset *= 2) {
        uint val = 0;
        if (lid >= offset) {
            val = temp[lid - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid >= offset) {
            temp[lid] += val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < n) {
        out_arr[gid] = temp[lid];
    }
    
    if (lid == GROUP_SIZE - 1) {
        block_sums[group_id] = temp[lid];
    }
}