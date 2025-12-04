#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input,
    __global const uint* prefixSum,
    __global       uint* output,
                   uint n,
                   uint offset)
{
    const uint x = get_global_id(0);
    const uint localX = get_local_id(0);
    const uint val = ((x < n) ? input[x] : 0);

    const uint pattern = (val >> offset) & (RADIX - 1);

    __local uint localScan[RADIX][GROUP_SIZE];
    for (int i = 0; i < RADIX; ++i) {
        localScan[i][localX] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    localScan[pattern][localX] = 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int shift = 1; shift < GROUP_SIZE; shift *= 2) {
        uint buf[RADIX];

        for (int i = 0; i < RADIX; ++i) {
            if (localX >= shift) {
                buf[i] = localScan[i][localX - shift];
            } else {
                buf[i] = 0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < RADIX; ++i) {
            localScan[i][localX] += buf[i];
        }   

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x >= n) {
        return;
    }

    const uint groupId = get_group_id(0);
    const uint globalOffset = ((pattern > 0 || groupId > 0) ? 
        prefixSum[pattern * ((n + GROUP_SIZE - 1) / GROUP_SIZE) + groupId - 1] : 0);
    const uint localOffset = localScan[pattern][localX] - 1;

    // printf("group: %d\tval: %d\tglobalOffset: %d\tlocalOffset: %d\n",
    //     groupId, val, globalOffset, localOffset);

    output[globalOffset + localOffset] = val;
}