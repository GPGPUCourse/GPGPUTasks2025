#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global       uint* buffer2)
{
    const unsigned int local_id = get_local_id(0);
    
    __local uint prefixes[16];
    
    if (local_id < 16) {
        prefixes[local_id] = buffer1[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id == 0) {
        uint sum = 0;
        for (unsigned int i = 0; i < 16; i++) {
            uint temp = prefixes[i];
            prefixes[i] = sum;
            sum += temp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id < 16) {
        buffer2[local_id] = prefixes[local_id];
    }
}
