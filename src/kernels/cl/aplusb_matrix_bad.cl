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

    const unsigned int group_size_by_col = get_local_size(0);
    const unsigned int group_id_by_col = get_group_id(0);
    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);

    // Логика расписана для группы размеров 32 * n, но алгоритм подходит и для других размеров.
    
    // Покрываем первые 32 блока задачи.
    // Идея в том, чтобы взять локальные индексы группы [0..31] и назначить им
    // задачи из того блока, что соответствует локальному индексу. При этом первая группа будет покрывать всегда
    // первую задачу из каждого рассмотренного блока, вторая группа -- вторую и тд.
    // То есть получим, что первая группа с локальными индексами [0..31] покроет задачи [0, 32, 64, ... 992].
    // Вторая группа имеет те же локальные индексы, но так как она вторая, делаем смешение на вторую задачу в блоке,
    // т.е. покрываем [1, 33, 65, ... 993]
    const unsigned int shift_by_col = get_local_id(0) * group_size_by_col + group_id_by_col % group_size_by_col;

    // Так как блоков у нас больше 32, то повторим ту же логику для блоков с 33 по 64, с 65 по 96 ...
    // То есть сделаем дополнительный сдвиг.
    // У нас есть группы, каждая из которых содержит 32 блока. Нам нужно найти а в какую группу входит рассматриваемый
    // сейчас блок (например, 70-ый блок будет входить в 2 группу), умножить это на количество блоков (уже покрытых)
    // и умножить на количество задач в каждом блоке
    const unsigned int shift_by_blocks = (group_id_by_col / group_size_by_col) * group_size_by_col * group_size_by_col;

    const unsigned int idx = row * width + shift_by_col + shift_by_blocks;
    if (idx >= width * height)
        return;

    c[idx] = a[idx] + b[idx];
}
