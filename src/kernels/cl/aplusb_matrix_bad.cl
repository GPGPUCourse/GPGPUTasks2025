#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__kernel void aplusb_matrix_bad(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    // все три массива - линейно выложенные двумерные матрицы размера width (число столбиков) x height (число рядов)
    // при этом в памяти подряд идут элементы являющимися соседями в рамках одного ряда,
    // т.е. матрица выложена в памяти линейно ряд за рядом
    // т.е. если в матрице сделать шаг вправо или влево на одну ячейку - то в памяти мы шагнем на 4 байта
    // т.е. если в матрице сделать шаг вверх или вниз на одну ячейку - то в памяти мы шагнем на так называемый stride=width*4 байта

    // TODO реализуйте этот кернел - просуммируйте две матрицы так чтобы получить максимально ПЛОХУЮ производительность с точки зрения memory coalesced паттерна доступа
    const size_t gx  = get_global_id(0);
    const size_t gy  = get_global_id(1);
    const size_t gz  = get_global_id(2);
    const size_t gsx = get_global_size(0);
    const size_t gsy = max((size_t)1, get_global_size(1));
    const size_t gsz = max((size_t)1, get_global_size(2));
    const size_t gid = gx + gy * gsx + gz * gsx * gsy;
    const size_t N = (size_t)width * (size_t)height;
    if (gid >= N) return;
    const size_t row = gid % (size_t)height;
    const size_t col = gid / (size_t)height;
    const size_t idx = row * (size_t)width + col;
    c[idx] = a[idx] + b[idx];

}
