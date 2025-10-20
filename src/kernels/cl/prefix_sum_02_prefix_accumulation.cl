#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    uint global_idx = get_global_id(0);
    if (global_idx >= n) return;

    uint stride = 1u << pow2;
    uint full_pair_width = stride + stride;

    uint intra_pair_offset = global_idx & (full_pair_width - 1u);

    if (intra_pair_offset < stride) return;

    uint which_pair = global_idx >> (pow2 + 1u);
    uint left_sibling = which_pair << 1u;

    prefix_sum_accum[global_idx] += pow2_sum[left_sibling];
}
