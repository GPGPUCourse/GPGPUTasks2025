#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1)
{
    // TODO
    const unsigned int local_id = get_local_id(0);
    
    __local uint sums[16];
    __local uint temp[GROUP_SIZE];
    
    if (local_id < 16) {
        sums[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id < 16) {
        uint sum = 0;
        for (unsigned int wg = 0; wg < a1; wg++) {
            sum += buffer1[wg * 16 + local_id];
        }
        sums[local_id] = sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id < 16) {
        buffer2[local_id] = sums[local_id];
    }
}
