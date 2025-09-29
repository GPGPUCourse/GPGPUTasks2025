#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
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

    const uint global_x = get_global_id(0);
    const uint global_y = get_global_id(1);
    const uint global_size_x = get_global_size(0);
    const uint global_size_y = get_global_size(1);

    const ulong width64 = (ulong)width;
    const ulong height64 = (ulong)height;
    const ulong total_threads = (ulong)global_size_x * (ulong)global_size_y;
    const ulong total_elements = width64 * height64;
    const ulong linear_id = (ulong)global_y * (ulong)global_size_x + (ulong)global_x;

    if (linear_id >= total_elements)
        return;

    // Обрабатываем элементы в column-major порядке, чтобы соседние потоки
    // обращались к элементам памяти со stride = width, ухудшая coalescing.
    for (ulong cm_index = linear_id; cm_index < total_elements; cm_index += total_threads) {
        const ulong column = cm_index / height64;
        const ulong row    = cm_index - column * height64;

        if (column >= width64 || row >= height64)
            continue;

        const ulong index64 = row * width64 + column;
        const uint index = (uint)index64;
        c[index] = a[index] + b[index];
    }
}
