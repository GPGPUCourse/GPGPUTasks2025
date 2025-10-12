#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
        __global const uint* pow2_sum,         // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
        __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
        unsigned int n,
        unsigned int pow2)
{
    const unsigned int period = (1 << (pow2 + 1));
    const unsigned int in_period_offset = ((1 << pow2) - 1);
    const unsigned int i = get_global_id(0);
    const unsigned int index = (i / (1 << pow2)) * period + (in_period_offset + i % (1 << pow2));

    if (index >= n)
        return;

    uint current_prefix_sum = 0;
    if (pow2 != 0) {
        current_prefix_sum = prefix_sum_accum[index];
    }

    // let's say (index+1) = 2^k1 + 2^k2 + 2^k3 + ..., where is k1 > k2 > k3 > ...
    bool fast_finish = true;
    uint index_decomposition_future_sum = 0; // those 2^k1 + 2^k2 + ... , who already will be taken into account in future (with larger pow2)
    for (int k = 31; k >= 0; --k) {
        if ((index + 1) & (1 << k)) { // if we have 2^k in (index+1) decomposition
            if (pow2 == k) { // if currently in pow2_sum partial sums including part of our index decomposition
                rassert(index_decomposition_future_sum % (1 << pow2) == 0, 65754325);
                current_prefix_sum += pow2_sum[index_decomposition_future_sum / (1 << pow2)];
            }
            index_decomposition_future_sum += (1 << k);
        }
        if (fast_finish && pow2 == k) {
            break;
        }
    }
    if (!fast_finish) {
        rassert((index + 1) == index_decomposition_future_sum, 142341231);
    }

    prefix_sum_accum[index] = current_prefix_sum;
}
