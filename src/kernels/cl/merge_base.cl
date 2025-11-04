#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_base(
    __global const uint *input_data,
    __global uint *output_data,
    int k,
    int n)
{
    const uint index = get_global_id(0);
    if (index >= n)
    {
        return;
    }
    const uint value = input_data[index];
    const uint block = (index >> k);
    const uint left = (block) << k;
    const uint start = ((block >> 1) << 1) << k;
    uint l = (block ^ 1) << k;
    uint r = ((block ^ 1) << k) + (1 << k);
    if (l >= n)
    {
        output_data[index] = value;
        return;
    }
    if (r > n)
    {
        r = n;
    }
    for (uint i = l; i < r; ++i)
    {
        const uint x = input_data[i];
        if (x > value || (x == value && (block & 1) == 0))
        {
            output_data[start + (index - left) + (i - l)] = value;
            return;
        }
    }
    output_data[start + (index - left) + (r - l)] = value;
}