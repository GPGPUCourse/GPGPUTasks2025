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

     __local uint local_sum[GROUP_SIZE];
    
    const uint global_id = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);
    
    uint thread_sum = 0;
    for (uint k = 0; k < LOAD_K_VALUES_PER_ITEM; k++) {
        uint index = global_id * LOAD_K_VALUES_PER_ITEM + k;
        if (index < n) {
            thread_sum += a[index];
        }
    }
    
    local_sum[local_id] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (uint stride = GROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            local_sum[local_id] += local_sum[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        b[group_id] = local_sum[0];
    }
}
