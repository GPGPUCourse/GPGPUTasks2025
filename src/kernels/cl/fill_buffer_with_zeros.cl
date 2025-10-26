#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill_buffer_with_zeros(
    __global uint* input,
    __global uint* output,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    if (input == 0) {
        output[i] = 0;
    } else {
        output[i] = input[i];
    }
}
