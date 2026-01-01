#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_01_local_counting(
    __global const uint* buffer1,
    __global uint* buffer2,
    unsigned int n,
    unsigned int offset)
{
    // считает для buffer1[0, n) кол-во кейсов 0, 1, ..., 2^(RADIX_BITS) - 1 для offset
    // сохраняет в buffer2 [0, 2^RADIX_BITS) x [0, ceil(n / GROUP_SIZE))

    unsigned int idx = get_global_id(0);
    unsigned int local_idx = get_local_id(0);

    __local unsigned int workgroup_data[GROUP_SIZE];

    const unsigned int mask = (1 << RADIX_BITS) - 1;
    const unsigned int NO_DATA = 1 << RADIX_BITS;

    if (idx >= n)
        workgroup_data[local_idx] = NO_DATA;
    else
        workgroup_data[local_idx] = (buffer1[idx] >> offset) & mask;

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int output_size_Y = (n + GROUP_SIZE - 1) / GROUP_SIZE;

    if (local_idx == 0) {
        unsigned int result[1 << RADIX_BITS];
        for (int i = 0; i < (1 << RADIX_BITS); i++)
            result[i] = 0;

        for (int i = 0; i < GROUP_SIZE; i++) {
            if (workgroup_data[i] == NO_DATA)
                continue;
            result[workgroup_data[i]]++;
        }

        for (int i = 0; i < (1 << RADIX_BITS); i++)
            buffer2[i * output_size_Y + (idx / GROUP_SIZE)] = result[i];
    }
}
