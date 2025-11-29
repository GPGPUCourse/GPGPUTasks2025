#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global const uint* buffer2,
    __global uint* buffer3,
    int i_bit,
    unsigned int n)
{
    const uint index = get_global_id(0);
    if (index >= n)
        return;
    int is_one = (buffer1[index] >> i_bit) & 1;
    if (is_one == 0) {
        buffer3[index - buffer2[index]] = buffer1[index];
    } else {
        buffer3[n - buffer2[n - 1] + buffer2[index] - 1] = buffer1[index];
    }
}