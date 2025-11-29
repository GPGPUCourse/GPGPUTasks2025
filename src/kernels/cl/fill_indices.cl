#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
fill_indices(__global uint *indices, unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    indices[index] = index;
}