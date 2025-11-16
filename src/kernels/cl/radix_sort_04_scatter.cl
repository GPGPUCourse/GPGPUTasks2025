#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global const uint* group_sum_prefix,
    __global uint* output,
    unsigned int offset,
    unsigned int n)
{
    const uint gid = get_group_id(0);
    const uint lid = get_local_id(0);
    const uint idx = gid * GROUP_SIZE + lid;
    const uint n_groups = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    __local uint sum_prefix[16];
    __local uint offsets[16];
    __local uint buffer[GROUP_SIZE];

    if (lid < 16) {
        sum_prefix[lid] = 0;
        sum_prefix[lid] = (gid > 0) ? group_sum_prefix[(gid - 1) * 16 + lid] : 0;
        offsets[lid] = (lid > 0) ? group_sum_prefix[(n_groups - 1) * 16 + lid - 1] : 0;
    }
    buffer[lid] = idx < n ? input[idx] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
#pragma unroll
        for (uint i = 1; i < 16; ++i) {
            offsets[i] += offsets[i - 1];
        }

        const uint remaining = min((int)(n - gid * GROUP_SIZE), GROUP_SIZE);
        for (uint i = 0; i < remaining; ++i) {
            const uint mask = (16 - 1);
            const uint num = (buffer[i] >> offset) & mask;
            output[offsets[num] + sum_prefix[num]++] = buffer[i];
        }
    }
}