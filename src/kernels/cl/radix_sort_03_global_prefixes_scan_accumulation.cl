#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* source,
    __global       uint* digits,
    __global       uint* target,
    uint n,
    uint pw)
{
    uint global_id = get_global_id(0);
    if (global_id < n) {
        uint group_id = get_group_id(0);
        uint num_groups = get_num_groups(0);
        uint x = source[global_id];
        uint d = (x >> (8 * pw)) % RADIX;
        uint pos = atomic_inc(&digits[num_groups * d + group_id]);
        target[pos] = source[global_id];
    }
}
