#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* input,
    __global uint* output,
    unsigned int n,
    unsigned int offset)
{
    __local uint localA[RADIX][GROUP_SIZE];
    const uint x = get_global_id(0);
    const uint localX = get_local_id(0);

    for (int i = 0; i < RADIX; ++i) {
        localA[i][localX] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < n) {
        localA[(input[x] >> offset) & (RADIX - 1)][localX] = 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int shift = GROUP_SIZE / 2; shift > 0; shift /= 2) {
        if (localX < shift) {
            for (int i = 0; i < RADIX; ++i) {
                localA[i][localX] += localA[i][localX + shift];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localX < RADIX) {
        const uint groupId = get_group_id(0);
        // printf("groupId: %d\tpattern: %d\tcnt: %d\t\tidx: %d\n", 
        //     groupId, localX, localA[localX][0], localX * RADIX + groupId);
        output[localX * ((n + GROUP_SIZE - 1) / GROUP_SIZE) + groupId] = localA[localX][0];
    }
}