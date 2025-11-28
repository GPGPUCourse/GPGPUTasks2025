#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* last_iter,
    __global       uint* new_iter,
    const uint n)
{

    const uint i = get_global_id(0) << 1;
    if (i < n) {
        unsigned int s = last_iter[i];
        if (i + 1 < n) {
            s += last_iter[i + 1];
        }
        new_iter[i >> 1] = s;
    }

}
