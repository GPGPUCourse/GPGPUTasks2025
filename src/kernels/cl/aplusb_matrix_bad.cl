#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__kernel void aplusb_matrix_bad(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    const unsigned int index_x = get_global_id(0);
    const unsigned int index_y = get_global_id(1);

    if (index_x >= width || index_y >= height) {
        return;
    }

    int correct_index = index_y * width + index_x;

    uint val_a = 0, val_b = 0;

    for(int i = 0; i < 8; i++) {
        int huge_stride = (i * 1024 + correct_index * 17) % (width * height);
        int random_offset = (correct_index * 31 + i * 127) % (width * height);

        val_a += a[(correct_index + huge_stride) % (width * height)];
        val_b += b[(correct_index + random_offset) % (width * height)];

        int transpose_x = index_x % height;
        int transpose_y = index_y % width;
        int transpose_idx = transpose_x * width + transpose_y;
        if(transpose_idx < width * height) {
            val_a += a[transpose_idx];
            val_b += b[transpose_idx];
        }
    }

    c[correct_index] = a[correct_index] + b[correct_index];

    // все три массива - линейно выложенные двумерные матрицы размера width (число столбиков) x height (число рядов)
    // при этом в памяти подряд идут элементы являющимися соседями в рамках одного ряда,
    // т.е. матрица выложена в памяти линейно ряд за рядом
    // т.е. если в матрице сделать шаг вправо или влево на одну ячейку - то в памяти мы шагнем на 4 байта
    // т.е. если в матрице сделать шаг вверх или вниз на одну ячейку - то в памяти мы шагнем на так называемый stride=width*4 байта

    // TODO реализуйте этот кернел - просуммируйте две матрицы так чтобы получить максимально ПЛОХУЮ производительность с точки зрения memory coalesced паттерна доступа
}
