#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void fill_buffer_with_zeros(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global uint* buffer1,
    // __global uint* buffer2,
    // __global uint* buffer3,
    unsigned int n,
    __global const uint* input
    )
{
    uint i = get_global_id (0);
    if (i >= n) return;
    buffer1[i] = input[i];
    // buffer2[i] = 0;
    // buffer3[i] = 0;
}
