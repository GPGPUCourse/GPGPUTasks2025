#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer_in,
    __global       uint* buffer_zeros,
    unsigned int antihype, // n
    unsigned int bit)
{
    const uint idx = get_global_id(0);
    if (idx >= antihype) {
        return;
    }

    buffer_zeros[idx] = ((buffer_in[idx] & (1u << bit)) == 0) ? 1u : 0u;
}
