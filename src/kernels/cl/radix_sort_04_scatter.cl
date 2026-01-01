#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    __global const uint* buffer1,
    __global const uint* buffer2,
    __global uint* buffer3,
    unsigned int n,
    unsigned int offset)
{
    // делает Radix Sort по RADIX_BITS битам с offset
    // в buffer3 [0, n)
    // buffer1 [0, n) - оригинальный массив
    // buffer2 [0, 2^RADIX_BITS) x [0, ceil(n / GROUP_SIZE)) - global offset

    unsigned int idx = get_global_id(0);
    unsigned int local_idx = get_local_id(0);

    __local unsigned int workgroup_data[GROUP_SIZE];
    __local unsigned int local_offset[1 << RADIX_BITS][GROUP_SIZE];

    unsigned int mask = (1 << RADIX_BITS) - 1;

    for (int i = 0; i < (1 << RADIX_BITS); i++)
        local_offset[i][(local_idx + 1) % GROUP_SIZE] = 0;

    unsigned int current;

    if (idx >= n) {
        workgroup_data[local_idx] = 123;
        current = 123;
    } else {
        workgroup_data[local_idx] = buffer1[idx];

        current = (workgroup_data[local_idx] >> offset) & mask;

        if (local_idx + 1 < GROUP_SIZE)
            local_offset[current][local_idx + 1] = 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_idx == 0) {

        // for (int j = 0; j < GROUP_SIZE; j++)
        //     printf("%d ", workgroup_data[j]);
        // printf("\n");

        // for (int i = 0; i < (1 << RADIX_BITS); i++) {
        //     for (int j = 0; j < GROUP_SIZE; j++) {
        //         printf("%d ", local_offset[i][j]);
        //     }
        //     printf("\n");
        // }
    }

    rassert(GROUP_SIZE >= (1 << RADIX_BITS), 53410);

    if (local_idx < (1 << RADIX_BITS)) {
        for (int i = 0; i < GROUP_SIZE; i++) {
            if (i > 0)
                local_offset[local_idx][i] += local_offset[local_idx][i - 1];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx >= n)
        return;

    const unsigned int buffer2_size_Y = ((n + GROUP_SIZE - 1) / GROUP_SIZE);

    unsigned int new_idx = local_offset[current][local_idx] + buffer2[current * buffer2_size_Y + (idx / GROUP_SIZE)];

    // printf("idx: %d, local_offset: %d, global_offset: %d\n", idx, local_offset[current][local_idx], buffer2[current * buffer2_size_Y + (idx / GROUP_SIZE)]);

    buffer3[new_idx] = workgroup_data[local_idx];
}