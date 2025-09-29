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
    const unsigned int index_x = get_global_id(0);
    const unsigned int index_y = get_global_id(1);

    if (index_x >= width || index_y >= height)
        return;

    const unsigned int index = index_x * height + index_y;

    c[index] = a[index] + b[index];
}
