#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain n values
    unsigned int n,
    const uint offset) // number of iteration
{
    const uint global_idx = get_global_id(0);
    if (global_idx >= n) return;

    uint v = pow2_sum[global_idx];
    if (global_idx >= offset){
        v += pow2_sum[global_idx - offset];
    }
    next_pow2_sum[global_idx] = v;
}
