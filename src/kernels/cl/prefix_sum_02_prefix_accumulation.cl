#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum1, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global const uint* pow2_sum2, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    uint pow2_1,
    unsigned int pow2_2)
{
    const uint i = get_global_id(0);
    const uint i_shift_1 = (i >> pow2_1);
    const uint i_shift_2 = (i >> pow2_2);

    if (i >= n) {
        return;
    }

    uint sum = 0;

    if (i_shift_1 & 1) {
        sum += pow2_sum1[i_shift_1 - 1];
    }

    if (i_shift_2 & 1) {
        sum += pow2_sum2[i_shift_2 - 1];
    }

    prefix_sum_accum[i] += sum;
}
