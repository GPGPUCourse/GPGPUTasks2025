#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer_pow_sum, // powers of 2
    __global       uint* buffer_pow_sum_next, // next powers of 2
    unsigned int n)
{
    const uint idx = get_global_id(0);
    const uint base = idx << 1;
    
    uint accum = 0;
    if (base < n) {
        accum = buffer_pow_sum[base];
    }
    if (base + 1 < n) {
        accum += buffer_pow_sum[base + 1];
    }
    buffer_pow_sum_next[idx] = accum;
}
