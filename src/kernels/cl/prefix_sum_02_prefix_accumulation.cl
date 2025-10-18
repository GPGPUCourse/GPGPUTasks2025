#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* buffer, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int buf_size)
{
    uint gid = get_global_id(0);
    uint buf_ind = gid + buf_size;
    uint accum = 0;
    while (buf_ind > 1) {
        if (gid % 2) {
            accum += buffer[buf_ind - 1];
        }
        buf_ind /= 2;
        gid /= 2;
    }
    prefix_sum_accum[get_global_id(0)] = accum;
}
