#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    __global const uint* pow2_sum, // contains n values
    __global uint* next_pow2_sum, // will contain (n+1)/2 values
    unsigned int n)
{
    size_t index = get_global_id(0);
    if (index < (n + 1) / 2) {
        uint sum = pow2_sum[2 * index];
        if (2 * index + 1 < n) {
            sum += pow2_sum[2 * index + 1];
        }
        next_pow2_sum[index] = sum;
    }
}
