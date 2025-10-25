#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input,
    __global const uint* pref_sum_big,
    __global       uint* output,
    unsigned int n,
    unsigned int pow2)
{
    __local int local_data[GROUP_SIZE];
    __local int pref_sum[GROUP_SIZE];
    uint group_id = get_group_id(0);
    uint i = get_global_id(0);
    uint local_i = get_local_id(0);

    if (i < n) {
        local_data[local_i] = input[i];
    } else {
        local_data[local_i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_i == 0) {
        pref_sum[0] = (local_data[0] >> pow2) & 1;
        for (uint index = 1; index < GROUP_SIZE; ++index) {
            pref_sum[index] = pref_sum[index - 1] + ((local_data[index] >> pow2) & 1);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < n) {
        uint sorted_index = pref_sum[local_i];
        if (group_id > 0) {
            sorted_index += pref_sum_big[group_id - 1];
        }

        if (((local_data[local_i] >> pow2) & 1) == 0) {
            output[i - sorted_index] = local_data[local_i];
        } else {
            uint count_zeros = n - pref_sum_big[(n - 1) / GROUP_SIZE];

            output[count_zeros + sorted_index - 1] = local_data[local_i];
        }
    }
}