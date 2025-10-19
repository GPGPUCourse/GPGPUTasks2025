#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    unsigned int x = get_global_id(0);
    uint n1 = 0, n2 = 0;
    if (2 * x < n) {
        n1 = pow2_sum[2 * x];
    }
    if (2 * x + 1 < n) {
        n2 = pow2_sum[2 * x + 1];
    }
    if (x > (n + 1) / 2) {
        return;
    }
    next_pow2_sum[x] = n1 + n2;
}
