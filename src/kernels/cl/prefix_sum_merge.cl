#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_merge(
    __global const uint* as, // input array
    __global const uint* group_prefix_sum, // group_prefix_sum[i] = sum[0; GROUP_SIZE * i]
    __global       uint* prefix_sum, // we want to make it finally so that prefix_sum[i] = sum[0, i]
    unsigned int n)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    __local uint local_as[GROUP_SIZE];
    local_as[lid] = 0;
    if (gid < n) {
        local_as[lid] = as[gid];
    }
    
    uint accum = local_as[lid];
    if (lid % 2) {
        accum += local_as[lid - 1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    accum += group_prefix_sum[get_group_id(0)];

    for (uint pow2 = 1; (1 << pow2) <= GROUP_SIZE; pow2++) {
        uint block_size = 1 << pow2;
        uint pair_sum = 0;
        if (lid < GROUP_SIZE / block_size) {
            pair_sum = local_as[2 * lid] + local_as[2 * lid + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < GROUP_SIZE / block_size) {
            local_as[lid] = pair_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        if ((lid / block_size) % 2) {
            accum += local_as[lid / block_size - 1];
        }
    }

    if (gid < n) {
        prefix_sum[gid] = accum;
    }
}
