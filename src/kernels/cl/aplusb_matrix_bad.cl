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
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint i = (y * width + x) / height;
    const uint j = (y * width + x) % height;
    const uint index = j * width + i;
    c[index] = a[index] + b[index];
}
