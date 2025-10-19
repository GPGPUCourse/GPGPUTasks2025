#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_inplace_sparse(
    __global uint* in,
    unsigned int n,
    uint level)
{
    // no barriers because work items do not intersect
    size_t i = get_global_id(0);
    size_t work_idx = (i + 1) * 2 * level - 1;
    if (work_idx < n) {
        size_t prev = (2 * i + 1) * level - 1;
        in[work_idx] += in[prev];
    }
}
