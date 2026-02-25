#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global uint* b,
                                    unsigned int  n)
{
    const uint id = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    __local uint local_buf[GROUP_SIZE];

    local_buf[lid] = (id < n) ? a[id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint step = GROUP_SIZE / 2; step > 0; step /= 2) {
        if (lid < step) {
            local_buf[lid] += local_buf[lid + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        b[gid] = local_buf[0];
    }
}