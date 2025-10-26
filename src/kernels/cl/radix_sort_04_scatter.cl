#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_04_scatter(
    __global const uint *input,
    __global const uint *prefix,
    __global uint *output,
    unsigned int n,
    unsigned int from,
    unsigned int ones)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    if (index < n)
    {
        const uint value = input[index];
        const uint x = (value >> from) & 1;
        const uint zeros = n - ones;
        // Also from previous HW
        uint i = index + 1;
        uint sum = 0;
        while (i > 0)
        {
            sum += prefix[i - 1];
            i = (i & (i - 1));
        }
        if (x)
        {
            // printf("place for %d: %d(%d - 1 + %d)\n", value, sum - 1 + zeros, sum, zeros);
            output[sum - 1 + zeros] = value;
        }
        else
        {
            // printf("place for %d: %d\n", value, index - sum);
            output[index - sum] = value;
        }
    }
}