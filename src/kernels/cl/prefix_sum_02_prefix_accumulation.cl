#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; (i+1)*2^pow2)
    __global uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    const uint i = get_global_id(0);
    if (i & (1 << pow2)) {
        const uint pow_i = (i >> (pow2 + 1)) << 1; // if k = 2: 0bABCDE -> 0bAB0, where A,B,C,D,E - binary digits (ex. 0b01101 -> 0b010)
        prefix_sum_accum[i] += pow2_sum[pow_i];
    }
}
