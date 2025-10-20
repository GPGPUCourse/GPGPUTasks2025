#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    __global       uint* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    const unsigned int index = get_global_id(0);
    if (index > n) {
        return;
    }
    const unsigned int prefix_index = (index + 1) / pow2;
//    if (index == 1) {
//        for (int i = 0; i < 10; ++i) {
//            printf("pow2_sum[%d] = %d\n", i, pow2_sum[i]);
//        }
//        printf("%d\n", prefix_index);
//        printf("\n");
//
//    }
    if (prefix_index % 2 == 1) {
        prefix_sum_accum[index] += pow2_sum[prefix_index - 1];
    }

}
