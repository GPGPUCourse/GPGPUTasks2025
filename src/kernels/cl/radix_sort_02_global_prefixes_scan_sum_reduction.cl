#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    uint n)
{
    const uint global_id = get_global_id(0);
    const uint num_outputs = (n + 1) >> 1;

    if (global_id < num_outputs) {
        uint in_base = global_id << 1;
        uint left_value = pow2_sum[in_base];
        uint right_value = 0;

        uint right_pos = in_base + 1;
        if (right_pos < n) {
            right_value = pow2_sum[right_pos];
        }

        next_pow2_sum[global_id] = left_value + right_value;
    }
}