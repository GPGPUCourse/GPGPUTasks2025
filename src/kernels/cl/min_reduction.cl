#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__kernel void min_reduction(
    __global const float* input,
    __global float* output,
    const uint n)
{
    const uint i = get_global_id(0);
    const uint localI = get_local_id(0);

    __local float localBuf[GROUP_SIZE];

    if (i < n) {
        localBuf[localI] = input[i];
    } else {
        localBuf[localI] = FLT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint shift = GROUP_SIZE / 2; shift > 0; shift >>= 1) {
        if (localI < shift) {
            localBuf[localI] = fmin(localBuf[localI], localBuf[localI + shift]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localI == 0) {
        output[get_group_id(0)] = localBuf[0];
    }
}