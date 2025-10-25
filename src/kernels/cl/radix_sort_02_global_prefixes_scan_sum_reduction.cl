#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global uint* buffer1,
    uint n, uint k
    )
{
    uint index = get_global_id(0);
    if (index >= n) {
        return;
    }
    uint size = 1 << k;
    uint begin = (index >> k) << (k + 1);
    uint res = begin + size + (index % size);
    if (res < n) {
        buffer1[res] += buffer1[begin + size - 1];
    }
}
