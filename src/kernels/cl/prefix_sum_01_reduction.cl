#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    const unsigned int out_size = (n + 1) / 2;

    if (i >= out_size) return;

    unsigned int idx1 = 2 * i;
    unsigned int idx2 = 2 * i + 1;

    if (idx2 < n) {
        next_pow2_sum[i] = pow2_sum[idx1] + pow2_sum[idx2];
    } else {
        next_pow2_sum[i] = pow2_sum[idx1];
    }
}
