#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(1, GROUP_SIZE, 1)))
__kernel void aplusb_matrix_bad(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= width || j >= height)
        return;

    // все три массива - линейно выложенные двумерные матрицы размера width (число столбиков) x height (число рядов)
    // при этом в памяти подряд идут элементы являющимися соседями в рамках одного ряда,
    // т.е. матрица выложена в памяти линейно ряд за рядом
    // т.е. если в матрице сделать шаг вправо или влево на одну ячейку - то в памяти мы шагнем на 4 байта
    // т.е. если в матрице сделать шаг вверх или вниз на одну ячейку - то в памяти мы шагнем на так называемый stride=width*4 байта

    // TODO реализуйте этот кернел - просуммируйте две матрицы так чтобы получить максимально ПЛОХУЮ производительность с точки зрения memory coalesced паттерна доступа
    // unsigned int id = j * width + i;

    i += (j * 1103515245 + 12345);
    i %= width;

    c[j * width + i] = a[j * width + i] + b[j * width + i];

    // i выбирается псевдослучайно с сидом j и конгруэнтным генератором как в glibc (Было 1.2 ГБ/с, стало 1 ГБ/с, потому что,
    //  как я понимаю, изначальный кернел уже на каждый элемент загружал по кэшлинии, что подтверждается тем, что отношение 166 / 1.2 слегка похоже на 128).
}
