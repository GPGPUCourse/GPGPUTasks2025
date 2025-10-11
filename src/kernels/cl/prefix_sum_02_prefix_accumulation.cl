#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* dual,
    __global       uint* pref,
    unsigned int size,
    unsigned int step)
{
    const unsigned int index = get_global_id(0);

    if (index >= size / step || index % 2 != 0) {
        return;
    }

    for (unsigned int i = 0; i < step; ++i) {
        const unsigned int offset = (index + 1) * step - 1 + i;

        if (offset >= size) {
            return;
        }

        pref[offset] += dual[index];
    }
}
