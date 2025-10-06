#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void mandelbrot(__global float* results,
                     unsigned int width, unsigned int height,
                     float fromX, float fromY,
                     float sizeX, float sizeY,
                     unsigned int iters, unsigned int isSmoothing)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= width || j >= height)
        return;

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    unsigned int iter = 0;

    while (iter < iters && (x * x + y * y) < threshold2)
    {
        float x1 = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = x1;
        iter++;
    }

    if (isSmoothing && iter != iters)
    {
        results[i + j * width] = iter - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    results[j * width + i] = 1.0f * iter / iters;
}
