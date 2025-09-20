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
    const unsigned int index_x = get_global_id(0);
    const unsigned int index_y = get_global_id(1);
    if (index_x >= width || index_y >= height)
        return;


    const unsigned int index = index_y * width + index_x;
    // На примере матрицы width = 5 и height = 3: 0 -> 0, 1 -> 5, 2 -> 10, 3 -> 1, 4 -> 6, 5 -> 11 и тд
    // Это соотвествие индексов, где при переходе берем следующий соседний(правее) элемент в столбце(или переходим на новую строку), к индексам, где при переходе берем элемент в соседней(ниже) строке
    // Можно заметить, что строка элемента, в который переходит индекс, это остаток от деления на height
    // Столбец, в который должен перейти элемент, это по сути количество элементов в его строке до него, то есть количество элементов с таким же остатком при делении на height, которые были до него
    // Таким образом, новые индексы(как для матрица)
    const unsigned int i = index % height;
    const unsigned int j = index / height;
    const unsigned int new_index = i * width + j;
    c[new_index] = a[new_index] + b[new_index];
}
