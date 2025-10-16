#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// не работает без синхронизации по всем рабочим группам
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_fused(
    __global const uint* pow2_sum, // contains curN values
    __global       uint* next_pow2_sum, // will contain (curN+1)/2 values
    __global       uint* next_pow2_sum_even,
    __global       uint* prefix_sum_accum,
    unsigned int curN,
    unsigned int n, // work range must be at least (n + 1) / 2
    unsigned int pow2) 
{
    const uint x = get_global_id(0);
    uint mx = (curN + 1) / 2 / 2;
    if (x < (curN + 1) / 2) {
        const uint sum = pow2_sum[x * 2] + (x * 2 + 1 < curN ? pow2_sum[x * 2 + 1] : 0);
        next_pow2_sum[x] = sum;
        if (x % 2 == 0) {
            mx = max(mx, x / 2);
            next_pow2_sum_even[x / 2] = sum;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    const uint range = (1 << pow2); 
    const uint idx = x / range * (range << 1) + range - 1 + (x % range);  

    if (idx < n) {
        // if (idx % 2 == 0 && pow2 == 1) {
        //     printf("x: %d\tidx: %d\tx/range: %d\tpow2: %d\tsum: %d\tadd: %d\n", 
        //         x, idx, x / range, pow2, prefix_sum_accum[idx], next_pow2_sum_even[x / range]);
        // }
        // rassert(x / range <= mx, 84756873);
        // if (x / range <= mx) {
        //     printf("wtf x/range: %d\tmx: %d\n", x / range, mx);
        // }
        prefix_sum_accum[idx] += next_pow2_sum_even[x / range];
    }
    // printf("x: %d   sum: %d   pow2_sum[%d]: %d   pow2_sum[%d]:%d\n", 
    //     x, next_pow2_sum[x], x * 2, pow2_sum[x * 2], x * 2 + 1, (x * 2 + 1 < n ? pow2_sum[x * 2 + 1] : 0));
}
