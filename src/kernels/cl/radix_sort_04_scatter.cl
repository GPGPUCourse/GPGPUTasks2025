#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input_buffer,
    __global const uint* prefix_sum_buffer,
    __global       uint* output_buffer,
    unsigned int n,
    unsigned int nk_width,
    unsigned int shift)
{
    __local uint buffer[GROUP_SIZE];

   unsigned int global_index = get_global_id(0);
    unsigned int local_index = get_local_id(0);
    unsigned int group_index = get_group_id(0);

    n -= (global_index - local_index);

    n = n > GROUP_SIZE ? GROUP_SIZE : n;

    if (local_index < n) {
        buffer[local_index] = input_buffer[global_index];
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int counting_row = local_index >> WORKITEM_MASK_BIT;
    unsigned int count = 0;
    if ((local_index & WORKITEM_MASK) == 0) {
        unsigned int s = 0;
        if (group_index + counting_row * nk_width > 0) {
            s = prefix_sum_buffer[group_index + counting_row * nk_width - 1];
        }

        for (unsigned int i = 0; i < n; ++i) {
            unsigned int tmp = buffer[i];
            if (((tmp >> shift) & COUNTING_MASK) == counting_row) {
                output_buffer[s++] = tmp;
            }
        }
    }
}