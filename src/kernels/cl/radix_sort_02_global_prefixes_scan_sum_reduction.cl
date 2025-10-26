#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
radix_sort_02_global_prefixes_scan_sum_reduction(
    __global uint *arr,
    unsigned int n,
    unsigned int k)
{
    // My code from previous HW
    __local uint local_data[GROUP_SIZE];
    const uint local_index = get_local_id(0);
    const uint index = get_global_id(0) * (1 << k) + (1 << k) - 1;
    if (index < n)
    {
        local_data[local_index] = arr[index];
    }
    else
    {
        local_data[local_index] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = 1; (1 << i) <= GROUP_SIZE; ++i)
    {
        if (((local_index + 1) & ((1 << i) - 1)) == 0)
        {
            local_data[local_index] += local_data[local_index - (1 << (i - 1))];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (index < n)
    {
        arr[index] = local_data[local_index];
    }
}
