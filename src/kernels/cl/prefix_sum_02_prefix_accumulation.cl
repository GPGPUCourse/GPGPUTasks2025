#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    __global const uint *pow2_sum,
    __global uint *prefix_sum_accum,
    unsigned int n)
{
    uint index = get_global_id(0) + 1;
    // const uint local_index = get_local_id(0);
    // __local uint add;
    // __local uint lower;
    // if (local_index == 0)
    // {
    //     lower = index - local_index;
    //     add = 0;
    //     uint x = (index - local_index) >> 8;
    //     while (x > 0)
    //     {
    //         add += pow2_sum[x - 1];
    //         x = (x & (x - 1));
    //     }
    // }
    if (index <= n)
    {
        uint sum = 0;
        while (index > 0)
        {
            sum += pow2_sum[index - 1];
            index = (index & (index - 1));
        }
        prefix_sum_accum[get_global_id(0)] = sum;
    }
}
