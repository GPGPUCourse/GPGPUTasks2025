#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global       uint* buckets,
    unsigned int n,
    unsigned int bit_start)
{
    // DONE

    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group = get_group_id(0);

    if (id < n) {
        unsigned int bucket = (input[id] >> bit_start) & (BUCKET_COUNT - 1);
        atomic_inc(&buckets[((n + GROUP_SIZE - 1) / GROUP_SIZE) * bucket + group]);
    }
}
