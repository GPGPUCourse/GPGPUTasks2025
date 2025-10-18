#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_sum_reduction(
    __global const uint* pow2_sum, // contains n values
    __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    const unsigned int index = get_global_id(0);
    const unsigned int out_size = (n + 1) / 2;
    
    if (index >= out_size)
        return;
    
    unsigned int idx0 = 2 * index;
    unsigned int idx1 = 2 * index + 1;
    
    uint sum = pow2_sum[idx0];
    if (idx1 < n) {
        sum += pow2_sum[idx1];
    }
    
    next_pow2_sum[index] = sum;
}
