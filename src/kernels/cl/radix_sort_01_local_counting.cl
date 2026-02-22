#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* buffer1,
    __global       uint* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    const size_t elems_per_block = 2u * GROUP_SIZE;
    const size_t group_id = get_group_id(0);
    const size_t local_id = get_local_id(0);
    const size_t base = group_id * elems_per_block;

    uint private_hist[4];
    private_hist[0] = 0u; 
    private_hist[1] = 0u; 
    private_hist[2] = 0u; 
    private_hist[3] = 0u;
    
    const size_t idx0 = base + local_id;
    const size_t idx1 = base + GROUP_SIZE + local_id;
    if (idx0 < a1) {
        uint digit = (buffer1[idx0] >> a2) & 3u;
        ++private_hist[digit];
    }
    if (idx1 < a1) {
        uint digit = (buffer1[idx1] >> a2) & 3u;
        ++private_hist[digit];
    }

    const size_t local_size = get_local_size(0);
    __local uint reduction_buffer[GROUP_SIZE];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint bits = 0; bits < 4; ++bits)
    {
        reduction_buffer[local_id] = private_hist[bits];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = local_size / 2; stride > 0; stride /= 2)
        {
            if (local_id < stride)
            {
                reduction_buffer[local_id] += reduction_buffer[local_id + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (local_id == 0u)
        {
            buffer2[4 * group_id + bits] = reduction_buffer[0];
        }
    }
}
