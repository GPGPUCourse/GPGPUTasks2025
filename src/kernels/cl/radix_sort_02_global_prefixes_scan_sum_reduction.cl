#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int n)
{
    unsigned int i = get_global_id(0);
    
    if (i >= (n + 1) / 2) {
        return;
    }
    
    uint left = pow2_sum[2 * i];
    uint right = (2 * i + 1 < n) ? pow2_sum[2 * i + 1] : 0u;
    
    next_pow2_sum[i] = left + right;
}
