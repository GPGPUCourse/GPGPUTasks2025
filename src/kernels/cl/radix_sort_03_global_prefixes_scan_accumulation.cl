#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* blocks,
    __global       uint* prefix,
    const unsigned int cnt_size,
    unsigned int cur_bucket)
{
    const uint group_id = get_group_id(0);
    const uint local_index = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint index = group_id * local_size + local_index;

    const uint base_prefix = cur_bucket * cnt_size;
    const uint base = cur_bucket * get_num_groups(0);

    uint to_add = blocks[base + group_id];
    if (index < cnt_size) {
        prefix[base_prefix + index] += to_add;
    }
}
