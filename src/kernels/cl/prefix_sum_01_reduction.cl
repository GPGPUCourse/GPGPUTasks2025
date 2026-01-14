#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* block_sums, // contains num_blocks values
    __global       uint* next_block_sums, // will contain (num_blocks+1)/2 values
    unsigned int num_blocks)
{
    const unsigned int gid = get_global_id(0);
    unsigned int i0 = gid * 2;
    unsigned int i1 = i0 + 1;
    if (i0 >= num_blocks) return;
    unsigned int sum = block_sums[i0];
    if (i1 < num_blocks) sum += block_sums[i1];
    next_block_sums[gid] = sum;
}
