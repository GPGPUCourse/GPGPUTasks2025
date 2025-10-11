#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* reduced, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int bit_to_check)
{
    unsigned const int pos = get_global_id(0);
    unsigned const pos_from_1 = pos + 1;

    if (pos >= n) return;

    if ((pos_from_1 >> bit_to_check) & 1) {
        unsigned const int block_size_power = bit_to_check + 1;
        unsigned const int block_number = pos_from_1 >> block_size_power;
        unsigned const int block_index = (block_number << 1);

        prefix_sum_accum[pos] += reduced[block_index];
    }
}
