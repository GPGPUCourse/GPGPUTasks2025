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
    const size_t x = get_global_id(0);
    const size_t y = get_global_id(1);
    const size_t xs = get_global_size(0);

    const size_t gid = y * xs + x;
    const size_t N = width * height;
    if (gid >= N){
         return;
    }

    const size_t i = (gid % height) * width + gid / height;

    c[i] = a[i] + b[i];
}
