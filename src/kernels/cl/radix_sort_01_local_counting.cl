#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* last_iter,
    __global       uint* new_iter,
    const uint n,
    const uint bit_to_sort)
{

    const uint i = get_global_id(0);
    if (i < n) {
        new_iter[i] = (last_iter[i] >> bit_to_sort) & 1;
    }

}
