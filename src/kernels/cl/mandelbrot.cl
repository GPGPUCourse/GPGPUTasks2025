#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// Сделал через float2/dot/length но не увидел сильного прироста производительности
// Но код стал немного лаконичнее
__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void mandelbrot(__global float* results,
                     unsigned int width, unsigned int height,
                     float fromX, float fromY,
                     float sizeX, float sizeY,
                     unsigned int iters, unsigned int isSmoothing)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    float2 z0 = (float2) (
		fromX + (i + 0.5f) * sizeX / width,
		fromY + (j + 0.5f) * sizeY / height
	);

	float2 z = z0;

    int iter = 0;
    for (; iter < iters && dot(z, z) < threshold2; ++iter) {
		z = (float2) (z.x * z.x - z.y * z.y, 2.0f * z.x * z.y) + z0;
    }
    float result = iter;
    if (isSmoothing != 0 && iter != iters) {
        result = result - log(log(length(z)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iters;
    results[j * width + i] = result;
}
