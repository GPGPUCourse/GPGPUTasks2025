#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void make_exclusive(
    __global const uint* input,
    __global uint* output,
    unsigned int n)
{
    unsigned int global_id = get_global_id(0);
    if (global_id >= n) return;
    if (global_id == 0u) output[global_id] = 0u;
    else output[global_id] = input[global_id - 1u];
}
