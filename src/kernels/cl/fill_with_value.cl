#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
fill_with_value(
    __global uint* data,
    uint value,
    uint n)
{
    const uint global_index = get_global_id(0);
    if (global_index < n) {
        data[global_index] = value;
    }
}
