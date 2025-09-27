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

    // чтобы добиться наихудшей эффективности надо чтобы мы брали элементы как то долго. то есть для каждого операции сложения мы ожидаем что элементы будут браться не из соседних ячеек памяти,
    // а из ячеек памяти которые далеко друг от друга лежат. в терминологии coalescing эффективно считается если два потока (один элемент матрицы = 1 поток) стоят рядом в памяти, то есть
    // в кеш линии. Если же два потока далеко друг от друга, то они не попадают в одну кеш линию и приходится делать два обращения к памяти вместо одного и это снижает производительность.

    const unsigned int index = get_global_id(0);
    const unsigned int n = width * height; // общее число элементов в матрице

    if (index >= n)
        return;

    const unsigned int row = index / width; // номер ряда
    const unsigned int col = index % width; // номер столбика

    // нужен новый индекс (плохой) чтобы неэффективно сложить
    const unsigned int bad_index = col * height + row;

    c[bad_index] = a[bad_index] + b[bad_index];

}
