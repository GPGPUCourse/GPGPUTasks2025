#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global const uint* prefix_sum_zeros,
    __global uint* output,
    unsigned int n,
    unsigned int bit)
{
    const uint ind = get_global_id(0);

    if (ind >= n) {
        return;
    }

    if (((input[ind] >> bit) & 1) == 0) {
        output[prefix_sum_zeros[ind] - 1] = input[ind];
    } else {
        uint total_num_of_zeros = prefix_sum_zeros[n - 1];
        uint zeros_before = prefix_sum_zeros[ind];
        uint offset = total_num_of_zeros - zeros_before;

        output[offset + ind] = input[ind];
    }

}