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
    __global const uint* prefix_sum_accum,
    __global uint* output,
    unsigned int n,
    unsigned int offset)
{
    const uint index = get_global_id(0);

    if (index >= n)
        return;

    const uint inp_bit = input[index];

    const bool last_bit = ((input[index] >> offset) & 1) == 1;

    uint local_offset, global_offset;
    if (last_bit) {
        local_offset = index - prefix_sum_accum[index];
        global_offset = prefix_sum_accum[n - 1];

        output[local_offset + global_offset] = inp_bit;
        return;
    }

    local_offset = 0;
    global_offset = prefix_sum_accum[index] - 1;
    output[local_offset + global_offset] = inp_bit;
}