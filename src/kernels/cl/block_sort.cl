#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void block_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  n)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_id = get_group_id(0);

    __local uint local_data[GROUP_SIZE];

    int global_idx = group_id * GROUP_SIZE + lid;
    if (global_idx < n) {
        local_data[lid] = input_data[global_idx];
    } else {
        local_data[lid] = UINT_MAX;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint size = 2; size <= GROUP_SIZE; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            uint pos = 2 * lid - (lid & (stride - 1));
            if (pos + stride < GROUP_SIZE) {
                uint a = local_data[pos];
                uint b = local_data[pos + stride];

                bool up = ((pos & size) == 0);
                if ((a > b) == up) {
                    local_data[pos] = b;
                    local_data[pos + stride] = a;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (global_idx < n) {
        output_data[global_idx] = local_data[lid];
    }
}
