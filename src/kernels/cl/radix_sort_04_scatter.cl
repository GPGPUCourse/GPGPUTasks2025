#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* buffer1,
    __global const uint* buffer2,
    __global       uint* buffer3,
    unsigned int a1,
    unsigned int a2)
{
    const size_t elems_per_block = 2u * GROUP_SIZE;
    const size_t group_id = get_group_id(0);
    const size_t local_id = get_local_id(0);
    const size_t base_idx = group_id * elems_per_block;

    const size_t idx0 = base_idx + local_id;
    const size_t idx1 = base_idx + GROUP_SIZE + local_id;
    const uint v0_valid = (idx0 < a1);
    const uint v1_valid = (idx1 < a1);
    const uint v0 = v0_valid ? buffer1[idx0] : 0u;
    const uint v1 = v1_valid ? buffer1[idx1] : 0u;
    const uint d0 = (v0 >> a2) & 3u;
    const uint d1 = (v1 >> a2) & 3u;

    uint base[4];
    base[0] = buffer2[4 * group_id + 0];
    base[1] = buffer2[4 * group_id + 1];
    base[2] = buffer2[4 * group_id + 2];
    base[3] = buffer2[4 * group_id + 3];

    __local uint s[2 * GROUP_SIZE];

    for (uint b = 0u; b < 4u; ++b) {
        s[local_id] = (v0_valid && d0 == b) ? 1u : 0u;
        s[local_id + GROUP_SIZE] = (v1_valid && d1 == b) ? 1u : 0u;
        barrier(CLK_LOCAL_MEM_FENCE);

        // upsweep
        for (size_t stride = 1; stride < elems_per_block; stride <<= 1) {
            size_t index = ((local_id + 1) * stride * 2) - 1;
            if (index < elems_per_block) {
                s[index] += s[index - stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (local_id == 0) {
            s[elems_per_block - 1] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // downsweep
        for (size_t stride = elems_per_block >> 1; stride >= 1; stride >>= 1) {
            size_t index = ((local_id + 1) * stride * 2) - 1;
            if (index < elems_per_block) {
                uint t = s[index - stride];
                s[index - stride] = s[index];
                s[index] += t;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (v0_valid && d0 == b) {
            buffer3[base[b] + s[local_id]] = v0;
        }
        if (v1_valid && d1 == b) {
            buffer3[base[b] + s[local_id + GROUP_SIZE]] = v1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}