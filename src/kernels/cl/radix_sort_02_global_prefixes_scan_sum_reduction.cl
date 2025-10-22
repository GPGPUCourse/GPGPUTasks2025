#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* from,
    __global uint* to,
    __global uint* prefix,
    const uint k,
    const uint compressed,
    const uint n)
{
    const uint index = get_global_id(0);
    if (index > n) {
        return;
    }

    const uint bin = index / compressed;
    const uint i = index % compressed;
    const uint index_offset = bin * compressed;
    const uint divided = (i + 1) >> k;
    prefix[index] = (k > 0 ? prefix[index] : 0) + ((divided & 1) ? from[index_offset + divided - 1] : 0);

    const uint next_1 = i << 1;
    const uint next_2 = next_1 + 1;
    const uint size = compressed >> k;
    to[index] = (next_1 < size ? from[index_offset + next_1] : 0) + (next_2 < size ? from[index_offset + next_2] : 0);
}
