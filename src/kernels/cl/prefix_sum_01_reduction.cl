#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void prefix_sum_01_sum_reduction(
    __global const uint* pow2_sum,
    __global  uint* next_pow2_sum,
    unsigned int n)
{
    uint out_n = (n + 1u) / 2u;
    for (uint i = 0; i < out_n; i++)
    {
        uint a = pow2_sum[2u*i];
        uint b = (2u*i + 1u < n) ? pow2_sum[2u*i + 1u] : 0u;
        next_pow2_sum[i] = a + b;
    }
}
