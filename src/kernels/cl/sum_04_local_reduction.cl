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
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_id = get_group_id(0);
    __local uint local_data[GROUP_SIZE];

    local_data[local_index] = (index < n) ? a[index] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if(local_index < stride) {
            local_data[local_index] += local_data[local_index + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_index == 0) b[group_id] = local_data[0];

}
