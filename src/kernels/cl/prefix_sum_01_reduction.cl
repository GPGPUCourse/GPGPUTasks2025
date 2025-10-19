#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global uint* buffer,
    unsigned int buf_size)
{
    uint lid = get_local_id(0);
    uint g_start = get_group_id(0) * GROUP_SIZE + buf_size;
    uint real_group_size = buf_size < GROUP_SIZE ? buf_size : GROUP_SIZE;
    __local uint local_buffer[GROUP_SIZE];
    local_buffer[lid] = 0;
    if (lid < real_group_size) {
        local_buffer[lid] = buffer[g_start + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint pow2 = 1; (1 << pow2) <= real_group_size; pow2++) {
        uint block_size = 1 << pow2;
        uint pair_sum = 0;
        if (lid < real_group_size / block_size) {
            pair_sum = local_buffer[2 * lid] + local_buffer[2 * lid + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < real_group_size / block_size) {
            local_buffer[lid] = pair_sum;
            buffer[g_start / block_size + lid] = pair_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}