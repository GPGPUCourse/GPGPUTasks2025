#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__kernel void aplusb_matrix_bad(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    const unsigned int index0 = get_global_id(0);
    const unsigned int index1 = get_global_id(1);
    if (index0 >= width || index1 >= height)
        return;

    const unsigned int index = index0 + index1 * width;
    c[index] = a[index] + b[index];
}
