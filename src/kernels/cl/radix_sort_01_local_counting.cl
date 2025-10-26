#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* array,
    __global       uint* digits,
    uint n,
    uint pow)
{
    uint global_id = get_global_id(0);
    if (global_id < n) {
        uint group_id = get_group_id(0);
        uint num_groups = get_num_groups(0);
        uint x = array[global_id];
        uint digit = (x >> (8 * pow)) % RADIX;
        atomic_inc(&digits[num_groups * digit + group_id]);
    }
}
