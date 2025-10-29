#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global uint* global_sums,
    __global uint* local_offsets,
    unsigned int numGroups
) {
    const unsigned int global_id = get_global_id(0);
    if (global_id == 0) {
        unsigned int s = 0;
        for (unsigned int i = 0; i < RADIX; ++i) {
            unsigned int tmp = global_sums[i];
            global_sums[i] = s;
            s += tmp;
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (global_id < RADIX) {
        unsigned int s = 0;
        unsigned int i = global_id * numGroups;
        for (unsigned int group = 0; group < numGroups; ++group) {
            unsigned int tmp = local_offsets[i + group];
            local_offsets[i + group] = s;
            s += tmp;
        }
    }
}