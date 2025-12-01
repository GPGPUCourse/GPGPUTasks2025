#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* in,
    __global const uint* num_of_zeroes,
    __global       uint* out,
    unsigned int n,
    unsigned int bit)
{
    const uint i = get_global_id(0);

    if(i >= n) {
        return;
    }

    uint idx = ((in[i] >> bit) & 1) == 0 ? num_of_zeroes[i] - 1 : num_of_zeroes[n - 1] + i - num_of_zeroes[i];
    out[idx] = in[i];

}