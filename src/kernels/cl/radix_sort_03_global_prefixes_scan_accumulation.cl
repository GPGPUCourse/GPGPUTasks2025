#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n,
    unsigned int m)
{
    unsigned int index = get_global_id(0);
    if (index >= n) {
        return;
    }

    prefix_sum_accum[index] = 0;

    unsigned int lci = 0;
    for (int i = m; i >= 0; --i) {
        unsigned int flag = ((index + 1) >> i) & 1;
        prefix_sum_accum[index] += pow2_sum[lci] * flag;
        lci = (lci << 1) + 1 + (2 * flag);
    }
}
