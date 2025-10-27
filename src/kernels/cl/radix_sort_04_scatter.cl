#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* flag_buffer,
    __global const uint* index_buffer,
    __global const uint* input_buffer,
    __global uint* output_buffer,
    unsigned int n,
    unsigned int bits_per_iter)
{
    int i = get_global_id(0);
    if (i >= n * bits_per_iter) {
        return;
    }
    if (flag_buffer[i]) {
        output_buffer[index_buffer[i] - 1] = input_buffer[i % n];
    }
}
