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
    if (i >= width || j >= height) {             // чтобы не было выхода за границы
        return;
    }
    const unsigned int index = j * width + i;    // линейный индекс пикселя в буфере и координаты
    const float x0 = fromX + (sizeX * i) / width;
    const float y0 = fromY + (sizeY * j) / height;
    float x = 0.0f;                              // начальное значение
    float y = 0.0f;
    unsigned int iteration = 0;
    while (x*x + y*y <= 4.0f && iteration < iters) {
        float xtemp = x*x - y*y + x0;            // z_{n+1}.x = x^2 - y^2 + x0
        y = 2.0f*x*y + y0;                       // z_{n+1}.y = 2xy + y0
        x = xtemp;
        iteration = iteration + 1;
    }
    float v = (iteration == iters)               // если точка в множестве то 0 (тк черное) иначе нормировка чтобы в диапзаоне была 0 1
        ? 0.0f
        : (float)iteration / (float)iters;
    results[index] = v;
}
