#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* input_buf,
    __global const uint* local_offsets,
    __global       uint* output_buf,
    const unsigned int n,
    const unsigned int shift)
{
    uint local_id = get_local_id(0);
    uint group_id = get_group_id(0);
    uint global_id = get_global_id(0);

    __local uint digits[GROUP_SIZE];

    uint valid = (global_id < n);
    uint value = valid ? input_buf[global_id] : 0;
    uint digit = (value >> shift) & RADIX_MASK;

    digits[local_id] = valid ? digit : 0xFFFFFFFF;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint rank = 0;
    if (valid) {
        for (uint i = 0; i < local_id; i++) {
            if (digits[i] == digit) rank++;
        }
    }

    if (valid) {
        uint base = local_offsets[group_id * NUM_BOXES + digit];
        output_buf[base + rank] = value;
    }
}
