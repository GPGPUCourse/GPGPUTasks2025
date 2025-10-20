#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);

    if (i < n && (i & 1)) {
        next_pow2_sum[i / 2] = pow2_sum[i] + pow2_sum[i - 1];
    }
}
