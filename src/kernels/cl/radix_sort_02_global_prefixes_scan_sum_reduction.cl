#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int n,
    unsigned int bitseq_len)
{
    uint id = get_global_id(0);
    uint i = id >> bitseq_len;
    uint seq = id & ((1u << bitseq_len) - 1);
    if (i * 2 < n) {
        next_pow2_sum[id] = pow2_sum[((i * 2) << bitseq_len) + seq];
    }
    if (i * 2 + 1 < n) {
        next_pow2_sum[id] += pow2_sum[((i * 2 + 1) << bitseq_len) + seq];
    }
}
