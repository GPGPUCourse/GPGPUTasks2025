#include "helpers/rassert.cl"
#include "../defines.h"

// actually copy of buffers is implemented herew instead of filling with zeros
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill_buffer_with_zeros(
    __global       uint* buffer1,
    __global const uint* buffer2,
    const unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i < n) buffer1[i] = buffer2[i];
}
