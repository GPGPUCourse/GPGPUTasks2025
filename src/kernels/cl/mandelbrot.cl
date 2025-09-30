#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void mandelbrot(__global float* results,
                     const unsigned int width, const unsigned int height,
                     const float fromX, const float fromY,
                     const float sizeX, const float sizeY,
                     const unsigned int iters, const unsigned int isSmoothing)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= width || j >= height) {
        return;
    }

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const float x0 = fromX + (i + 0.5f) * sizeX / width;
    const float y0 = fromY + (j + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        const float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }
    float result = iter;
    if (isSmoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iters;
    results[j * width + i] = result;
}
