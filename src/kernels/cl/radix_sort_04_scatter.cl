#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void radix_sort_04_scatter(
    __global const uint* offset,
    __global const uint* bit_buf,
    __global const uint* a,
    __global       uint* output,
    unsigned int n,
    __global const uint* previous_pref_sums)
{
    unsigned int i = get_global_id(0);
    unsigned int prev_offset = previous_pref_sums[n - 1];
    
    unsigned int to = bit_buf[i] * (offset[i] - 1 + prev_offset) + (bit_buf[i] == 0) * (previous_pref_sums[i] - 1);
    if (i < n && to < n) {
        // printf("%ld --> %ld'th = (%ld - 1 + %ld)\n", a[i], to, offset[i], prev_offset);
        output[to] = a[i];
    }
}