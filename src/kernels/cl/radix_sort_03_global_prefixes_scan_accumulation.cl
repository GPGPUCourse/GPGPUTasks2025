#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum,
    __global       uint* sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    size_t gid = get_global_id(0);
    size_t left_boundary = gid / (1ull << pow2);
    if (gid >= n || left_boundary == 0) return;
    if (!(gid & (1ull << pow2))) return;

    sum_accum[gid] += pow2_sum[left_boundary - 1];
}
