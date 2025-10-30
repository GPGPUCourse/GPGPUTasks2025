#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input_array,
    __global       uint* output_array,
    unsigned int n,
    unsigned int d) // at level d, add element from distance 2^d
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    const unsigned int offset = 1u << d; // 2^d

    if (i >= offset) {
        output_array[i] = input_array[i] + input_array[i - offset];
    } else {
        output_array[i] = input_array[i];
    }
}
