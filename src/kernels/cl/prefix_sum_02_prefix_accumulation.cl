#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    const unsigned int i = get_global_id(0);
    
    if (i < n && ((i + 1 >> pow2) & 1)) {
        prefix_sum_accum[i] += pow2_sum[(i + 1 >> pow2) - 1];
    }
}
