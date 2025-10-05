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

    // TODO
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];

    if (index >= n) {
        local_data[local_index] = 0;
    } else {
        local_data[local_index] = a[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index) {
        return;
    }

    uint temp = 0;
    for (uint idx = 0; idx < GROUP_SIZE; ++idx) {
        temp += local_data[idx];
    }
    b[get_group_id(0)] = temp;
}
