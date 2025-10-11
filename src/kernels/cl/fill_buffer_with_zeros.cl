#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void fill_buffer_with_zeros(
    __global uint* buffer,
    unsigned int n)
{
    const unsigned int global_id = get_global_id(0);
    if (global_id >= n) return;
    buffer[global_id] = 0;
}
