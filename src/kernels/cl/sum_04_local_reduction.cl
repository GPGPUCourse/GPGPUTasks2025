#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_04_local_reduction(__global const uint* a,
                                     __global       uint* b,
                                            unsigned int  n)
{
    // Подсказки:
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_id = get_group_id(0);
    __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    // TODO
    if (index < n) {
        local_data[local_index] = a[index];
    } else {
        local_data[local_index] = 0;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (uint i = GROUP_SIZE / 2; i > 0; i /= 2) {
        if (local_index < i) {
            local_data[local_index] += local_data[local_index + i];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_index == 0) {
        b[group_id] = local_data[0];
    }
}
