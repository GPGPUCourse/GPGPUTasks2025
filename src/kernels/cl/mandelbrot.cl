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

    if (i >= width || j >= height) return;

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    float x0 = fromX + ( (float)i + 0.5f ) * sizeX / (float)width;
    float y0 = fromY + ( (float)j + 0.5f ) * sizeY / (float)height;

    float x = x0;
    float y = y0;

    unsigned int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }

    float result = (float)iter;
    if (isSmoothing != 0u && iter != iters) {
        float mag = sqrt(x * x + y * y);
        result = result - log(log(mag) / log(threshold)) / log(2.0f);
    }

    result = result / (float)iters;
    results[j * width + i] = result;
}
