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
    __global       uint* cnt4,
    unsigned int n,
    unsigned int bit_offset)
{
    __local uint sum0[GROUP_SIZE];
    __local uint sum1[GROUP_SIZE];
    __local uint sum2[GROUP_SIZE];
    __local uint sum3[GROUP_SIZE];

    const uint group_id = get_group_id(0);
    const uint local_index = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint index = group_id * local_size + local_index;
    
    sum0[local_index] = 0u;
    sum1[local_index] = 0u;
    sum2[local_index] = 0u;
    sum3[local_index] = 0u;
    if (index < n) {
        uint cur_bit = (input[index] >> bit_offset) & 3u;
        sum0[local_index] = (cur_bit == 0u);
        sum1[local_index] = (cur_bit == 1u);
        sum2[local_index] = (cur_bit == 2u);
        sum3[local_index] = (cur_bit == 3u);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint j = local_size >> 1; j > 0; j >>= 1) {
        if (local_index < j) {
            sum0[local_index] += sum0[local_index + j];
            sum1[local_index] += sum1[local_index + j];
            sum2[local_index] += sum2[local_index + j];
            sum3[local_index] += sum3[local_index + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        const uint offset = get_num_groups(0) * get_num_groups(1);
        cnt4[group_id] = sum0[0];
        cnt4[offset + group_id] = sum1[0];
        cnt4[2u * offset + group_id] = sum2[0];
        cnt4[3u * offset + group_id] = sum3[0];
    }
}
