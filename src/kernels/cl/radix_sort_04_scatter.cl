#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global const uint* buffer2,
    __global const uint* buffer3,
    __global       uint* buffer4,
    unsigned int a1,
    unsigned int a2,
    unsigned int a3)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int work_group_id = get_group_id(0);
    
    __local uint offsets[16];
    
    if (local_id < 16) {
        uint offset = buffer3[local_id];
        for (unsigned int wg = 0; wg < work_group_id; wg++) {
            offset += buffer2[wg * 16 + local_id];
        }
        offsets[local_id] = offset;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    unsigned int idx = work_group_id * GROUP_SIZE + local_id;
    if (idx < a1) {
        uint value = buffer1[idx];
        unsigned int digit = (value >> a2) & 0xF;
        unsigned int pos = atomic_inc(&offsets[digit]);
        buffer4[pos] = value;
    }
}