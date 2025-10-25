#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* array,
    __global const uint* prefix_sums,
    __global       uint* result,
    unsigned int n,
    unsigned int bit_start)
{
    // DONE

    const unsigned int id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group = get_group_id(0);
    
    __local unsigned int local_pref[GROUP_SIZE * BUCKET_COUNT];
    __local unsigned int bucket_pref[BUCKET_COUNT];

    if (local_id == 0) {
        unsigned int bucket = (array[group * GROUP_SIZE] >> bit_start) & BUCKET_MASK;
        for (int i = 0; i < BUCKET_COUNT; ++i) {
            local_pref[i * GROUP_SIZE] = (bucket == i ? 1 : 0);
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int i = 1; i < GROUP_SIZE; i++) {
        if (i + group * GROUP_SIZE < n && local_id < BUCKET_COUNT) {
            const unsigned int bucket = (array[i + group * GROUP_SIZE] >> bit_start) & BUCKET_MASK;

            local_pref[local_id * GROUP_SIZE + i] = local_pref[local_id * GROUP_SIZE + i - 1];
            if (bucket == local_id) {
                ++local_pref[local_id * GROUP_SIZE + i];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (id < n) {
        const unsigned int bucket = (array[id] >> bit_start) & BUCKET_MASK;

        unsigned int global_pref = prefix_sums[bucket * ((n + GROUP_SIZE - 1) / GROUP_SIZE) + group - 1];

        unsigned int local_pref_add = 0;
        if (local_id > 0) {
            local_pref_add = local_pref[bucket * GROUP_SIZE + local_id - 1];
        }

        result[global_pref + local_pref_add] = array[id];
    }
}
