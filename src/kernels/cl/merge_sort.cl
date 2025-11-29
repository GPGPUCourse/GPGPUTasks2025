#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
             const uint sorted_k,
             const uint n)
{
    const uint i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const uint left = i - (i % (sorted_k * 2));
    const uint mid = min(left + sorted_k, n);
    const uint right = min(left + sorted_k * 2, n);
    uint l = (i & sorted_k) ? left : mid;
    uint r = (i & sorted_k) ? mid : right;
    --l;
    const uint x = input_data[i] + (i >= mid);
    while(r - l > 1) {
        const uint m = (l + r) >> 1;
        const uint c = input_data[m] < x;
        l = l * (1 - c) + m * c;
        r = m * (1 - c) + r * c;
    }
    output_data[i + r - mid] = input_data[i];
}
