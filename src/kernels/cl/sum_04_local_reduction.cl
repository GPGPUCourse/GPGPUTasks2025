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
    bool valid_index = (index < n / LOAD_K_VALUES_PER_ITEM);

    uint item_sum = 0;
    for (int i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
        // We have to set zeros to local memory for invalid indexes anyway
        item_sum += a[i * (n/LOAD_K_VALUES_PER_ITEM) + index] * valid_index;
    }

    __local uint local_data[GROUP_SIZE];
    const uint local_index = get_local_id(0);

    local_data[local_index] = item_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        uint wg_sum = 0;
        
        for (int i = 0; i < GROUP_SIZE; ++i) {
            wg_sum += local_data[i];
        }

        b[get_group_id(0)] = wg_sum;
    }
}
