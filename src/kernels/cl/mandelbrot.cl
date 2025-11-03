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

    if (i >= width || j >= height) {
        return;
    }

    const unsigned int idx = j * width + i

    const float threshold = 4.0f;

    float x0 = fromX + ((float) i / (float) width) * sizeX;
    float y0 = fromY + ((float) j / (float) height) * sizeY;

    float x = 0.0f;
    float y = 0.0f;
    unsigned int n = 0;

    while (n < iters && (x*x + y*y) <= threshold) {
        float x2 = x * x;
        float y2 = y * y;
        float xy = x * y;

        x = xx - yy + x0;
        y = 2.0f * xy - y0;
        ++n;
    }

    float value;
    if (isSmoothing && n < iters) {
        float r2 = x * x + y * y;
        if (r2 <= 0.0f) {
            value = (float) n / (float) iters;
        } else {
            float r = sqrt(r2);
            float nu = log2(log2(r));
            float smoothIter = (float) n + 1.0f - nu;
            value = smoothIter / (float) iters;
        } else {
            value = (float) n / (float) iters;
        }
    }

    results[idx] = value;
}
