#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    // если в левом блоке, то считаю количество антихайпов < текущего элемента из правого блока + текущий индекс в блоке
    // если в правом блоке, то считаю количество антихайпов справа <= текущего элемента из левого блока + текущий индекс в блоке
    // ура, я инплейс мердж двух блоков

    int block_id = i / sorted_k;
    int start_blocks = i - i % sorted_k;
    if (block_id % 2 == 1) {
        start_blocks -= sorted_k;
    }

    if (block_id % 2 == 0) {
        // левый

        int l = 0;
        int r = sorted_k;

        while (r - l > 1) {
            int mid = (l + r) / 2;

            if (mid + start_blocks + sorted_k >= n) {
                r = mid;
                continue;
            }

            if (input_data[mid + start_blocks + sorted_k] < input_data[i]) {
                l = mid;
            } else {
                r = mid;
            }
        }

        if (start_blocks + sorted_k < n && input_data[start_blocks + sorted_k] < input_data[i]) {
            ++l;
        }
        output_data[l + i] = input_data[i];
    } else {
        // правый

        int l = 0;
        int r = sorted_k;

        while (r - l > 1) {
            int mid = (l + r) / 2;

            if (input_data[mid + start_blocks] <= input_data[i]) {
                l = mid;
            } else {
                r = mid;
            }
        }

        if (input_data[start_blocks] <= input_data[i]) {
            ++l;
        }

        int number_of_less = l + (i - start_blocks - sorted_k);
        output_data[number_of_less + start_blocks] = input_data[i];
    }
}
