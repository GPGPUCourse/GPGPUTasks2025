#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_02_prefix_accumulation(
    __global uint* a,
    __global uint* buffer,
    const int n,
    const unsigned int block_len
)
{
    unsigned int i = get_global_id(0);
    unsigned int from = i / block_len;
    if (i < n)
        a[i] += buffer[from];
}
