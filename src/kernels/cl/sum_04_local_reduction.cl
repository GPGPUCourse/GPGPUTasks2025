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
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];

    if (index < n) {
        local_data[local_index] = a[index];
    }
    else {
        local_data[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // количество активных потоков
    for (int i = GROUP_SIZE/2; i > 0; i = i/2) {
        if (local_index < i) {
            local_data[local_index] += local_data[i + local_index];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        b[get_group_id(0)] = local_data[0];
    }

    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);
}
