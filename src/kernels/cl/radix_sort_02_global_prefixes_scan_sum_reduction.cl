#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum,
    __global uint* pow2_next_sum,
    unsigned int prev_size,
    unsigned int n
    )
{
    const uint gid = get_global_id(0);
    if (gid >= n) return;

    const uint i = 2 * gid;
    const uint j = 2 * gid + 1;

    pow2_next_sum[gid] = (j < prev_size)
                         ? pow2_sum[i] + pow2_sum[j]
                         : pow2_sum[i];
}
