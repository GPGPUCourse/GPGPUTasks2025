#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void radix_sort_01_local_counting(
    __global const uint* a,
    __global       uint* bit_buf,
    __global       uint* bit_buf_inv,
    unsigned int bit_num,
    unsigned int n)
{
    unsigned int i = get_global_id(0);
    if (i < n) {
        bit_buf[i] = (a[i] >> bit_num) & 1;
        bit_buf_inv[i] = ~(a[i] >> bit_num) & 1;
    }
}
