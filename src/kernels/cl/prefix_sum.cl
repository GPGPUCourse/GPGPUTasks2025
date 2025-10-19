#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum(
    __global uint* buf,
    unsigned int n,
    unsigned int k)
{
    unsigned int half_fragment_size = 1 << (k - 1);
    unsigned int i = get_global_id(0);

    unsigned int fragment_base = (i >> (k - 1)) << k;
    unsigned int fragment_add = buf[fragment_base + half_fragment_size - 1];

    unsigned int fragment_index = fragment_base + half_fragment_size + (i % half_fragment_size);

    if (fragment_index < n)
        buf[fragment_index] += fragment_add;
}