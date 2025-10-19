#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    __global const uint* in,
    __global uint* reduced,
    unsigned int n)
{
    size_t global_idx = get_global_id(0);
    size_t local_idx = get_local_id(0);
    __local uint buf[GROUP_SIZE];
    if (global_idx < n) {
        buf[local_idx] = in[global_idx];
    } else {
        buf[local_idx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_idx == 0) {
        uint sum = 0;
        for (int i = 0; i < GROUP_SIZE; ++i) {
            sum += buf[i];
        }
        reduced[global_idx / GROUP_SIZE] = sum;
    }
}
