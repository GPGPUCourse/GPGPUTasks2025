#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global uint* data,
    unsigned int in,
    unsigned int out,
    unsigned int n)
{
    uint i = get_global_id(0);
    uint out_n = (n + 1) / 2;

    if (i >= out_n) {
        return;
    }

    uint fst = 2 * i;
    uint snd = 2 * i + 1;

    if (snd < n) {
        data[out + i] = data[in + fst] + data[in + snd];
    } else {
        data[out + i] = data[in + fst];
    }
}
