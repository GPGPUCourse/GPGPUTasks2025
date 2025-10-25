#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int n,
    unsigned int pow2)
{
    __local int local_data[GROUP_SIZE];
    uint group_id = get_group_id(0);
    uint i = get_global_id(0);
    uint local_i = get_local_id(0);

    if (i < n) {
        local_data[local_i] = (buffer1[i] >> pow2) & 1;
    } else {
        local_data[local_i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_i == 0) {
        uint sum = 0;
        for (uint index = 0; index < GROUP_SIZE; ++index) {
            sum += local_data[index];
        }
        buffer2[group_id] = sum;
    }
}
