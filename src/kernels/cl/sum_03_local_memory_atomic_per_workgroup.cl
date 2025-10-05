#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    const uint i = get_global_id(0);
    const uint local_index = get_local_id(0);
    
    __local uint local_data[GROUP_SIZE];

    if (i < n) {
        local_data[local_index] = a[i];
    } else {
        local_data[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        uint local_sum = 0;
        for (uint j = 0; j < GROUP_SIZE; ++j) {
            local_sum += local_data[j];
        }
        atomic_add(sum, local_sum);
    }

}
