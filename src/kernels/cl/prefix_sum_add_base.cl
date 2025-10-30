#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define BLOCK_SIZE 512

__kernel void prefix_sum_add_base(
    __global       uint* data,
    __global const uint* block_prefix,
    unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int bid = get_group_id(0);

    if (gid >= n) return;

    if (bid > 0) {
        data[gid] += block_prefix[bid - 1];
    }
}
