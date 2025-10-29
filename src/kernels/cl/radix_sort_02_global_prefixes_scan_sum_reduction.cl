#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* local_hist,
    __global       uint* global_sums,
    unsigned int numGroups
) {
    const unsigned int global_id = get_global_id(0);
    if (global_id >= RADIX) return;

    unsigned int s = 0;
    for (unsigned int i = 0; i < numGroups; ++i) {
        s += local_hist[global_id * numGroups + i];
    }
    global_sums[global_id] = s;
}