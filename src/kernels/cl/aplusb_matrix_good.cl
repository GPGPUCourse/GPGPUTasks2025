#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__kernel void aplusb_matrix_good(__global const uint* a,
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

    // TODO реализуйте этот кернел - просуммируйте две матрицы так чтобы получить максимально ХОРОШУЮ производительность с точки зрения memory coalesced паттерна доступа

    unsigned int y = get_global_id(1) * 256;
    // const unsigned int local_row = get_local_id(1);
    const unsigned int x = get_global_id(0);
    if (x >= width || y >= height) {
        return;
    }
    // printf("x: %d   y: %d   id0: %d   id1: %d   local_id0: %d   local_id1: %d\n", 
    //     x, y, get_global_id(0), get_global_id(1), get_local_id(0), local_row);
    // printf("global_size0: %d   global_size1: %d\n", get_global_size(0), get_global_size(1));
    // printf("local_size0: %d   local_size1: %d\n", get_local_size(0), get_local_size(1));
    const unsigned int yBound = min(y + 256, height);
    // get_global_offset()
    // printf("moving down\n");
    for (; y < yBound; ++y) {
        int idx = y * width + x;
        c[idx] = a[idx] + b[idx];
        // printf("x: %d  y: %d  idx: %d  a[idx]: %d  b[idx]: %d  c[idx]: %d\n", 
        //     x, y, idx, a[idx], b[idx], c[idx]);
    }
}
