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
    __global const uint* prefmap,
    __global       uint* output,
    unsigned int n,
    unsigned int pos)
{
    const uint i = get_global_id(0);

    if (i >= n) {
        return;
    }

    uint shift = input[i] >> pos;
    uint start = i > 0 ? prefmap[i - 1] : 0;
    
    uint j;
    if (shift & 1) {
        uint total_zeros = n - prefmap[n - 1];
        j = total_zeros + start;
    } else {
        j = i - start;
    }

    output[j] = input[i];
}    