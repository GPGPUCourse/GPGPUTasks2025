#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_03_add_input(
    __global const uint* input,
    __global       uint* prefix_sum_accum,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    prefix_sum_accum[i] += input[i];
}
