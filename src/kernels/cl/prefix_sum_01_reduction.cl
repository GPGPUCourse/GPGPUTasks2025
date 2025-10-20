#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* input, // contains n values
    __global       uint* output, // will contain (n+1)/2 values
    unsigned int n)
{
    unsigned int i = get_global_id(0);
    unsigned int out_n = (n + 1) / 2;
    
    if (i < out_n) {
        uint left = input[i * 2];
        uint right = (i * 2 + 1 < n) ? input[i * 2 + 1] : 0;
        output[i] = left + right;
    }
}
