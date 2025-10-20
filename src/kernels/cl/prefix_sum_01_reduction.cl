#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    uint global_new_index = get_global_id(0);

    if (global_new_index >= n / 2) {
        return;
    }

    uint global_old_index = global_new_index * 2;
    uint global_old_index_2 = global_old_index + 1;

    if (global_old_index < n && global_old_index_2 < n) {
        next_pow2_sum[global_new_index] = pow2_sum[global_old_index] + pow2_sum[global_old_index_2];
    } else if (global_old_index < n) {
        next_pow2_sum[global_new_index] = pow2_sum[global_old_index];
    }
}
