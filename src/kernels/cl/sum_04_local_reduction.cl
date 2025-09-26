#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32
#define REDUCTION 4

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
    if (index < n) {
        local_data[local_index] = a[index];
    } else {
        local_data[local_index] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint sum = 0;
    uint count = GROUP_SIZE;
    while (count > 1) {
        sum = 0;
        bool master = local_index % REDUCTION == 0 && local_index < count;
        if (master) {
            uint end = min(count, local_index + REDUCTION);
            for (uint i = local_index; i < end; i++) {
                sum += local_data[i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (master) {
            local_data[local_index / REDUCTION] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        count = (count + REDUCTION - 1) / REDUCTION;
    }

    if (local_index == 0) {
        b[index / GROUP_SIZE] = local_data[0];
    }
}
