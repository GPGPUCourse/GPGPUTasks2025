#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void mandelbrot(
    __global float * const results,
    const unsigned int width,
    const unsigned int height,
    const float fromX,
    const float fromY,
    const float sizeX,
    const float sizeY,
    const unsigned int iters)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= width || j >= height) {
        return;
    }

    const float threshold = 65536.0f;
    const float x0 = fromX + (i + 0.5f) * sizeX / width;
    const float y0 = fromY + (j + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    unsigned int iter = 0;
    for (; iter < iters; ++iter) {
        const float x_ = x;
        x = x * x - y * y + x0;
        y = 2.0f * x_ * y + y0;
        if ((x * x + y * y) > threshold) {
            break;
        }
    }

    results[j * width + i] = 1.0f * iter / iters;
}
