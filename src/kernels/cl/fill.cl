#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void fill(
    __global uint* buffer,
    unsigned int n,
    unsigned int c)
{
    const unsigned int i = get_global_id(0);

    if (i < n) {
        buffer[i] = c;
    }
}
