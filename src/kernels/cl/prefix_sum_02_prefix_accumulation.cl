#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global       uint* output,
    __global const uint* block_sums,
    unsigned int n,
    unsigned int block_size)
{
    const unsigned int gid = get_global_id(0);
    if (gid >= n) return;
    unsigned int block_id = gid / block_size;
    unsigned int val = input[gid];
    if (block_id > 0) {
        val += block_sums[block_id - 1];
    }
    output[gid] = val;
}
