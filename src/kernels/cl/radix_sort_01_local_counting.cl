#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* a,
    __global       uint* buckets,
    unsigned int n,
    unsigned int offset)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);

    __local uint local_buckets[BUCKET_SIZE];

    if (local_index < BUCKET_SIZE) {
        local_buckets[local_index] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (index >= n) {
        return;
    }

    const uint bucket = (a[index] >> offset) & BUCKET_MASK;

    atomic_add(&local_buckets[bucket], 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint group_idx = get_group_id(0);

    const uint num_groups = get_num_groups(0);

    // WARN: incorrect if GROUP_SIZE < 2^bucket_bits
    if (local_index < BUCKET_SIZE) {
        buckets[local_index * num_groups + group_idx] = local_buckets[local_index];
    }
}
