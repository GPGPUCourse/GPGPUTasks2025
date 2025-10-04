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

    // DONE

    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_index = get_group_id(0);
    __local uint local_data[GROUP_SIZE];

    const bool is_master_thread = (local_index == 0);

    if (index >= n || index < 0) {
        local_data[local_index] = 0;
    } else {
        local_data[local_index] = a[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (is_master_thread) {
        uint local_sum = 0;

        for (uint i = 0; i < GROUP_SIZE; ++i) {
            local_sum += local_data[i];
        }

        b[group_index] = local_sum;
    }
}
