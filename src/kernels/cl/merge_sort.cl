#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
    unsigned int  sorted_k, // длина отсортированных кусочков
    unsigned int  n) // длина массива
{
    const unsigned int i = get_global_id(0); // номер элемента

    if (i >= n) {
        return;
    }

    unsigned int left_bound = (i / (2 * sorted_k)) * (2 * sorted_k);
    unsigned int middle = min(left_bound + sorted_k, n);
    unsigned int right_bound = min(left_bound + 2 * sorted_k, n);

    if (i < middle) {
        unsigned int value = input_data[i];

        int l = 0, r = right_bound - middle;
        while (l < r) {
            int m = (l + r) >> 1;
            if (input_data[middle + m] <= value) {
                l = m + 1; 
            } else {
                r = m;
            }
        }
        output_data[left_bound + (i - left_bound) + l] = value;
    } else if (i < right_bound) {
        unsigned int value = input_data[i];
        int l = 0, r = middle - left_bound;
        while (l < r) {
            int m = (l + r) >> 1;
            if (input_data[left_bound + m] < value) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        output_data[left_bound + (i - middle) + l] = value;
    }
}
