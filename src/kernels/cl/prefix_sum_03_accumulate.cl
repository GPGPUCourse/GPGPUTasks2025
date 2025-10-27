#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_03_accumulate(
    __global const uint* in,
    __global const uint* sparse,
    __global uint* out,
    unsigned int n)
{
    size_t global_idx = get_global_id(0);
    size_t local_idx = get_local_id(0);
    if (global_idx < n) {
        uint pref_sum = 0;
        uint cur_idx = global_idx + 1;
        while (cur_idx > 0) {
            pref_sum += sparse[cur_idx - 1];
            cur_idx -= cur_idx & -cur_idx;
        }
        out[global_idx] = pref_sum;
    }
}
