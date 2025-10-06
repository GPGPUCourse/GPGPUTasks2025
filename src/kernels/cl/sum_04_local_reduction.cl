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
}
