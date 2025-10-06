#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sum_04_local_reduction(__global const uint* a,
    __global uint* b,
    unsigned int n)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint group_id = get_group_id(0);
    __local uint local_data[GROUP_SIZE];

    local_data[lid] = (gid < n) ? a[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint lsum = 0;
        for (int i = 0; i < GROUP_SIZE; ++i) {
            lsum += local_data[i];
        }
        b[group_id] = lsum;
    }
}
