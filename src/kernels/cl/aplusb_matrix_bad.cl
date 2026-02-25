#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__kernel void aplusb_matrix_bad(__global const uint* a,
                                __global const uint* b,
                                __global       uint* c,
                                unsigned int width,
                                unsigned int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int idx = x * height + y;
        c[idx] = a[idx] + b[idx];
    }
}