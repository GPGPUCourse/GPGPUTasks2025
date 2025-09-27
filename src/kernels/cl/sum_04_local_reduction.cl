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
    __local uint local_data[GROUP_SIZE];
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_index = get_group_id(0);


    if (index < n) {
        local_data[local_index] = a[index];
    } else {
        local_data[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
       uint res = 0;
       for (size_t i = 0; i < GROUP_SIZE; ++i) {
           res += local_data[i];
       }
       b[group_index] = res;
    }
}
