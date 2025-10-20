#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* input,
    __global       uint* output,
    unsigned int n,
    unsigned int offset)
{
    const unsigned int index = get_global_id(0);
    
    if (index >= n)
        return;
    
    uint value = input[index];
    if (index >= offset) {
        value += input[index - offset];
    }
    output[index] = value;
}
