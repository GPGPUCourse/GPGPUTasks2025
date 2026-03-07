#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* in,
    __global const uint* global_pref,
    __global       uint* out,
    unsigned int n,
    unsigned int offset)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);

    if (index >= n) {
        return;
    }

    const uint bucket = (in[index] >> offset) & BUCKET_MASK;

    __local uint pref_sums[BUCKET_SIZE * GROUP_SIZE];

    // __local uint buf1[BUCKET_SIZE * GROUP_SIZE];
    // __local uint buf2[BUCKET_SIZE * GROUP_SIZE];

    for (uint b = 0; b < BUCKET_SIZE; b++) {
        pref_sums[b * GROUP_SIZE + local_index] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    pref_sums[bucket * GROUP_SIZE + local_index] = 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        for (int b = 0; b < BUCKET_SIZE; b++) {
            for (uint i = 1; i < GROUP_SIZE; i++) {
                pref_sums[b * GROUP_SIZE + i] += pref_sums[b * GROUP_SIZE + i - 1];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // barrier(CLK_LOCAL_MEM_FENCE);

    // bool buf2_target = true;

    // buf2[bucket * GROUP_SIZE + local_index] = 1;
    // barrier(CLK_LOCAL_MEM_FENCE);

    // uint pow2 = 1;
    // const uint count = local_index + 1;
    // while (true) {
    //     if (pow2 & count) {
    //         for (uint b = 0; b < BUCKET_SIZE; b++) {
    //             if (buf2_target) {
    //                 pref_sums[b * GROUP_SIZE + local_index] += buf2[b * GROUP_SIZE + count / pow2 - 1];
    //             } else {
    //                 pref_sums[b * GROUP_SIZE + local_index] += buf1[b * GROUP_SIZE + count / pow2 - 1];
    //             }
    //         }
    //     }

    //     pow2 *= 2;
    //     if (pow2 > count) {
    //         break;
    //     }

    //     buf2_target = !buf2_target;

    //     // assume that GROUP_SIZE is power of 2
    //     if (local_index * pow2 < GROUP_SIZE) {
    //         for (uint b = 0; b < BUCKET_SIZE; b++) {
    //             if (buf2_target) {
    //                 buf2[b * GROUP_SIZE + local_index] = buf1[b * GROUP_SIZE + local_index * 2] + buf1[b * GROUP_SIZE + local_index * 2 + 1];
    //             } else {
    //                 buf1[b * GROUP_SIZE + local_index] = buf2[b * GROUP_SIZE + local_index * 2] + buf2[b * GROUP_SIZE + local_index * 2 + 1];
    //             }
    //         }
    //     }
    //     barrier(CLK_LOCAL_MEM_FENCE);
    // }

    // barrier(CLK_LOCAL_MEM_FENCE);
    uint group_id = get_group_id(0);
    uint num_groups = get_num_groups(0);
    uint global_pref_index = bucket * num_groups + group_id;
    uint new_index = 0;
    if (global_pref_index > 0) {
        new_index += global_pref[global_pref_index - 1];
    }
    new_index += pref_sums[bucket * GROUP_SIZE + local_index] - 1;

    out[new_index] = in[index];
}