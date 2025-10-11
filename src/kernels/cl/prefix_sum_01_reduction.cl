#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_01_reduction(
    __global const uint* source, // contains n values
    __global       uint* result, // will contain (n+1)/2 values
    unsigned int n)
{
    unsigned const int k = 2 * get_global_id(0);
    if (k >= n) return;
    unsigned const int current = source[k];
    unsigned const int next_pos = k + 1 < n ? k + 1 : 0;
    unsigned const int next = source[next_pos];
    result[k / 2] = current + next;
}
