#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
fill_buffer_with_zeros(
    __global uint* buffer)
{
    size_t index = get_global_id(0);
    buffer[index] = 0;
}
