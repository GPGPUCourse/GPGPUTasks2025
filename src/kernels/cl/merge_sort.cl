#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
    int  sorted_run, // длина сорт подотрезка
    int  n) // длина массива
{
    int global_index = get_global_id(0);
    if (global_index >= n)
        return;

    int block_size = 2 * sorted_run; // размер блока (2 куска по k)
    int block_start = (global_index / block_size) * block_size; // начало блока
    int index_in_block = global_index - block_start; // позиция внутри блока

    int left_begin = block_start; // старт левого куска
    int left_end = min(block_start + sorted_run, n); // конец левого куска
    int right_begin = left_end; // анологично
    int right_end = min(block_start + block_size, n);

    int left_len = left_end - left_begin; // длина левого
    int right_len = right_end - right_begin; // длина правого

    if (left_len == 0)
    {
        output_data[global_index] = input_data[right_begin + index_in_block]; // брать из правого
        return;                                          // выходим (не надо бин поиск делать)
    }
    if (right_len == 0)
    {
        output_data[global_index] = input_data[left_begin  + index_in_block];
        return;
    }

    // take_left + take_right = index_in_block
    int lo = max(0, index_in_block - right_len);// нижняя граница элементов из левого
    int hi = min(index_in_block, left_len); // верхняя граница элементов из левого

    while (lo < hi) // бинсерч (тк надо в мерджсорте нужны отсортированные куски)
    {
        int take_left = (lo + hi) >> 1; // скока то берем из левого - остальное из правого
        int take_right = index_in_block - take_left;

        // мало из левого -> сдвиг вправо
        if (take_left < left_len && take_right > 0 && input_data[left_begin  + take_left] < input_data[right_begin + take_right - 1])
        {
            lo = take_left + 1;
        } else if (take_right < right_len && take_left > 0 && input_data[right_begin + take_right] < input_data[left_begin + take_left - 1])
        {
            hi = take_left - 1;
        } else
        {
            lo = hi = take_left;
        }
    }

    int take_left  = lo; // скок идет из левой чаасти
    int take_right = index_in_block - take_left; // остальное что осталось после левой (после индекса)
    // если правый пустой значит надо брать слева иначе минимум из левого и правого
    bool choose_left = (take_right >= right_len) || (take_left < left_len && input_data[left_begin + take_left] <= input_data[right_begin + take_right]);
    output_data[global_index] = choose_left ? input_data[left_begin  + take_left]: input_data[right_begin + take_right]; // результат элемент который
    // будет означает значение одного места уже в фтнального массиве
}
