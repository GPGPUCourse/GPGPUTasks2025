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
    
    const unsigned int index1 = get_global_id(0);
    const unsigned int index2 = get_global_id(1);

    if (index1 >= width || index2 > height) {
        return;
    } 

    unsigned int matrix_index = index2 * width + index1;
    const unsigned int x = matrix_index % height;
    const unsigned int y = matrix_index / height;

    if (matrix_index % 2 == 0) {
        matrix_index = x * width + y;
    } else {
        matrix_index = ((height - x) % height) * width + (width - y) % width;
    }

    c[matrix_index] = a[matrix_index] + b[matrix_index];
}
