#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

uint calcNextSum(__global const uint* pow2_sum,
                 const uint curN,
                 const uint x) {
    if (x < (curN + 1) / 2) {
        return pow2_sum[x * 2] + (x * 2 + 1 < curN ? pow2_sum[x * 2 + 1] : 0);
    }
    return 0;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_another_fused(
    __global const uint* pow2_sum, // contains curN values
    __global       uint* next_pow2_sum, // will contain (curN+1)/2 values
    __global       uint* prefix_sum_accum,
    unsigned int curN,
    unsigned int n, // work range must be at least (n + 1) / 2
    unsigned int pow2) // at least 1
{
    const uint x = get_global_id(0);
    const uint range = (1 << pow2);
    const uint sumIdx = ((x >> pow2) << 1);
    const uint begIdx = sumIdx * range; + range - 1;  
    const uint idx = begIdx + (x % range);  

    uint sum0 = calcNextSum(pow2_sum, curN, sumIdx);
    uint sum1 = calcNextSum(pow2_sum, curN, sumIdx + 1);
    if (idx < n) {
        prefix_sum_accum[idx] += sum0;
    }

    if (x % range == 0) {
        next_pow2_sum[sumIdx] = sum0;
    }
    if (x % range == 1) {
        next_pow2_sum[sumIdx + 1] = sum1;
    }
}
