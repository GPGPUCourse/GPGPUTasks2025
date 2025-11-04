#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_fast(
    __global const uint *input_data,
    __global uint *output_data,
    int k,
    int n)
{
    const uint index = get_global_id(0);
    const uint block = (index >> k);
    int l = ((block ^ 1) << k) - 1;
    int r = ((block ^ 1) << k) + (1 << k);
    if (index >= n)
    {
        return;
    }

    const uint value = input_data[index];

    // printf("%d for %d\n", l, index);

    if (l >= n - 1)
    {
        output_data[index] = value;
        return;
    }
    const uint left = (block) << k;
    const uint start = ((block >> 1) << 1) << k;
    const int start_l = ((block ^ 1) << k);

    if (r > n)
    {
        r = n;
    }
    while (r - l > 1)
    {
        int m = (l + r) >> 1;
        uint x = input_data[m];
        if (x > value || (x == value && (block & 1) == 0))
        {
            r = m;
        }
        else
        {
            l = m;
        }
    }
    // printf("put %d to %d\n", value, (start + (index - left) + (r - start_l)));
    output_data[start + (index - left) + (r - start_l)] = value;
}