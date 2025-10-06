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

    // TODO
}
