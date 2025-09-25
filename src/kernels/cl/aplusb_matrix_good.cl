#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__kernel void aplusb_matrix_good(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
        const size_t x = get_global_id(0);
        const size_t y = get_global_id(1);
        if (x >= width || y >= height){
            return;
        }

        const size_t i = y * width + x;
        c[i] = a[i] + b[i];
}
