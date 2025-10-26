#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input,
    __global uint* output,
    const uint k,
    const uint n)
{
    const uint index = get_global_id(0);
    if (index >= n) {
        return;
    }
    const uint value = input[index];
    const uint size = 1 << k;
    const uint size_mask = size - 1;
    const uint block_mask = ~size_mask;
    const uint block_start = index & block_mask;
    const bool is_right = (index >> k) & 1;
    const uint other_start = block_start + (is_right ? -size : size);
    uint lr[2] = { other_start, other_start + size };
    #pragma unroll
    for (uint i = 0; i <= k; i++) {
        const uint m = (lr[0] + lr[1]) >> 1;
        const uint sample = m >= n ? (~0) : input[m];
        lr[sample > value || (sample == value && !is_right)] = m;
    }
    output[(block_start & (block_mask << 1)) + (lr[1] - other_start) + (index & size_mask)] = value;
}
