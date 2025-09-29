#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void aplusb_matrix_good(__global const uint* a,
    __global const uint* b,
    __global uint* c,
    unsigned int width,
    unsigned int height)
{
    // все три массива - линейно выложенные двумерные матрицы размера width (число столбиков) x height (число рядов)
    // при этом в памяти подряд идут элементы являющимися соседями в рамках одного ряда,
    // т.е. матрица выложена в памяти линейно ряд за рядом
    // т.е. если в матрице сделать шаг вправо или влево на одну ячейку - то в памяти мы шагнем на 4 байта
    // т.е. если в матрице сделать шаг вверх или вниз на одну ячейку - то в памяти мы шагнем на так называемый stride=width*4 байта

    const unsigned int index_x = get_global_id(0);
    const unsigned int index_y = get_global_id(1);
    const unsigned int index = index_x * width + index_y;
    
    if (index >= width * height) {
        return;
    }
    c[index] = a[index] + b[index];

}
