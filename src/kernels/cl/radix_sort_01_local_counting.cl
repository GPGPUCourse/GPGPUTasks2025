#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(__global const uint* a,
                                        __global       uint* sum,
                                        int i_bit,
                                        unsigned int  n)
{
    const uint index = get_global_id(0);
    if (index >= n)
        return;
    atomic_add(sum, (a[index] >> i_bit) & 1);
}
