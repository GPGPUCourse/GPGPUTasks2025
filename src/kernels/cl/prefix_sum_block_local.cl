#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define BLOCK_SIZE 512

__kernel void prefix_sum_block_local(
    __global const uint* input,
    __global       uint* output,
    __global       uint* block_sums,
    unsigned int n)
{
    __local uint temp[BLOCK_SIZE];

    const unsigned int tid = get_local_id(0);  // thread id within block (0..BLOCK_SIZE-1)
    const unsigned int bid = get_group_id(0);  // block id
    const unsigned int gid = get_global_id(0); // global thread id

    if (gid < n) {
        temp[tid] = input[gid];
    } else {
        temp[tid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        uint val = 0;
        if (tid >= offset) {
            val = temp[tid - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (tid >= offset) {
            temp[tid] += val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid < n) {
        output[gid] = temp[tid];
    }

    if (tid == 0) {
        block_sums[bid] = temp[BLOCK_SIZE - 1];
    }
}
