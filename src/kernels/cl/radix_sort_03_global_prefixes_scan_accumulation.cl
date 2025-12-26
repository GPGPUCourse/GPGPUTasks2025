#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    // Z. O. V.
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2, // хотим прибавить если требуется к i-ой префиксной сумме кусок длины 2^pow2
    int bit)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    uint flag = (i+1) & (1u << pow2);
    if (bit == -1){
        if (flag){
            prefix_sum_accum[i] += pow2_sum[((i+1)>>pow2)-1];
        }
    } else {
        if (flag){
            prefix_sum_accum[i] += (((pow2_sum[((i+1)>>pow2)-1]>>bit) & 1u) + 1) % 2;
        }
    }


}