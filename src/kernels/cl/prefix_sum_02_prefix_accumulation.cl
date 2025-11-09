#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum,
    __global uint* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    (void)pow2;
    uint acc = 0u;
    for (uint i = 0; i < n; i++)
    {
        acc += pow2_sum[i];
        prefix_sum_accum[i] = acc;
    }
}
