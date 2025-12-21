#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer_pow_sum, // powers of 2
    __global       uint* buffer_prefix_sum,
    unsigned int n,
    unsigned int pow) // current power of two
{
    const uint idx = get_global_id(0);
    if (idx >= n) {
        return;
    }

    const uint block = (idx + 1) >> pow;
    if (block & 1u) {
        buffer_prefix_sum[idx] += buffer_pow_sum[block - 1];
    }
}
