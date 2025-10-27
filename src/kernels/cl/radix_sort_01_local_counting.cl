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
    unsigned int a1,
    unsigned int a2)
{
    // TODO
    const unsigned int local_id = get_local_id(0);
    const unsigned int work_group_id = get_group_id(0);
    const unsigned int global_id = get_global_id(0);
    
    __local uint histogram[16];
    
    if (local_id < 16) {
        histogram[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (global_id < a1) {
        unsigned int digit = (buffer1[global_id] >> a2) & 0xF;
        atomic_inc(&histogram[digit]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id < 16) {
        buffer2[work_group_id * 16 + local_id] = histogram[local_id];
    }
}
