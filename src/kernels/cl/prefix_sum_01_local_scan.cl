#include "helpers/rassert.cl"
#include "../defines.h"

// произошел бунд
__kernel void prefix_sum_01_local_scan(
    __global int* data,
    __global int* block_sums,
    unsigned int n,
    int is_inclusive)
{
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_size = get_local_size(0);

    if (gid >= n) return;

    __local int local_array[MAX_GROUP_SIZE];
    int original_value = data[gid];
    local_array[lid] = original_value;
    barrier(CLK_LOCAL_MEM_FENCE);

    // up-pass
    for (unsigned int stride = 1; stride < local_size; stride *= 2) {
        unsigned int index = (lid + 1) * stride * 2 - 1;
        if (index < local_size) {
            local_array[index] += local_array[index - stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == local_size - 1) {
        block_sums[group_id] = local_array[local_size - 1];
        local_array[local_size - 1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // down-pass
    for (unsigned int stride = local_size / 2; stride > 0; stride /= 2) {
        unsigned int index_j = (lid + 1) * stride * 2 - 1;
        if (index_j < local_size) {
            int temp = local_array[index_j - stride];
            local_array[index_j - stride] = local_array[index_j];
            local_array[index_j] += temp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < n) {
        if (is_inclusive) {
            data[gid] = local_array[lid] + original_value;
        } else {
            data[gid] = local_array[lid];
        }
    }
}
