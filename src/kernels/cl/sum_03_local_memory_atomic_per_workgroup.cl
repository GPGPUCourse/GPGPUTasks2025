#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global uint* sum,
                                                       const unsigned int n)
{
    const uint id = get_global_id(0);
    const uint lid = get_local_id(0);

    __local uint local_buf[GROUP_SIZE];
    local_buf[lid] = (id < n) ? a[id] : 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint group_sum = 0;
        for (int i = 0; i < GROUP_SIZE; ++i) {
            group_sum += local_buf[i];
        }
        atomic_add(sum, group_sum);
    }
}