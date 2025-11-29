#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
fill_index(
    __global uint* index,
    uint n)
{
    const uint global_index = get_global_id(0);
    if (global_index < n) {
        index[global_index] = global_index;
    }
}
