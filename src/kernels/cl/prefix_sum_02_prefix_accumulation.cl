#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* sums,
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n)
{
    uint i = get_global_id(0) + 1;

    if (i > n) {
        return;
    }

    uint cur_i = i;
    uint res = 0;
    uint offset = 0;
    uint cur_n = n;
    for(uint p=0; p<32; ++p) {
        if((cur_i >> p) & 1) {
            res += sums[offset + ((cur_i - 1) >> p)];
            cur_i -= (1 << p);
        }
        offset += cur_n;
        cur_n = cur_n / 2 + cur_n % 2;
    }
    prefix_sum_accum[i - 1] = res;
}
