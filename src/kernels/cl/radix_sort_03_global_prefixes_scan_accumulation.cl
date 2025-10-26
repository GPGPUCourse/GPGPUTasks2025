#include "helpers/rassert.cl"
#include "../defines.h"


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global       uint* buffer1, // bigger buffer
    __global const uint* buffer2, // smaller buffer
    const unsigned int buf1_sz)
{
    __local unsigned int local_data[GROUP_SIZE];
    const unsigned int idx = get_global_id(0);
    const unsigned int local_idx = get_local_id(0);

    local_data[local_idx] = (idx < buf1_sz ? buffer1[idx] : 0u);
    if (local_idx & 16u) local_data[local_idx] += local_data[local_idx ^ 16u];
    const unsigned int base = idx >> WARP_LG;
    if (base > 0u) {
        local_data[local_idx] += buffer2[((base - 1u) << (WARP_LG - 1u)) + (local_idx & 15u)];
    }
    if (idx < buf1_sz) {
        buffer1[idx] = local_data[local_idx];
    }
}
