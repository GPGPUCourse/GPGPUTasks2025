#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* array,
    __global       uint* buckets,
    unsigned int n,
    unsigned int start)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int l_idx = get_local_id(0);
    __local unsigned int l_buckets[1 << SORT_BUCKET_SIZE];

    if (l_idx < (1 << SORT_BUCKET_SIZE)) {
        l_buckets[l_idx] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx < n) {
        atomic_add(&l_buckets[(array[idx] >> start) % (1 << SORT_BUCKET_SIZE)], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_idx < (1 << SORT_BUCKET_SIZE)) {
        const unsigned int index = l_idx * ((n + GROUP_SIZE - 1) / GROUP_SIZE) + get_group_id(0);
        buckets[index] = l_buckets[l_idx];
    }
}
