#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_01_reduction(
    __global const uint* a,
    __global       uint* reduced,
    unsigned int n)
{
    // lets now reduce x2 every time
    
    unsigned int i = get_global_id(0);
    unsigned int thread = get_local_id(0);
    if (i < n) {
        if ((thread & 1) == 0) {
            reduced[i / 2] = a[i] + (i + 1 < n ? a[i + 1] : 0);
        }
    }
}
