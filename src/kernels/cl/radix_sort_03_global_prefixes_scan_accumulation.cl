#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> 
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(__global       uint* arr, 
                                                              __global const uint* block_sums, 
                                                              unsigned int n) 
{
    uint gid = get_global_id(0);
    uint group_id = get_group_id(0);
    if (gid < n && group_id > 0) {
        arr[gid] += block_sums[group_id - 1];
    }
}