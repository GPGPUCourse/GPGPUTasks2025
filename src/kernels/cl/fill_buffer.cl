#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill_buffer(
    __global const uint* input_buffer,
    __global uint* buffer_pow2,
    unsigned int n,
    unsigned int k)
{

    unsigned int index = get_global_id(0);

    if (index >= 2 * k - 1) {
        return;
    }

    if (index < k - 1 || index >= n + k - 1) {
        buffer_pow2[index] = 0;
    } else {
        buffer_pow2[index] = input_buffer[index + 1 - k];
    }
}