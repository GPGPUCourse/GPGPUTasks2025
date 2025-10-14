#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void copy_array(
    __global const uint* a,
    __global       uint* b, 
    unsigned int n)
{
    const uint x = get_global_id(0);
    if (x < n) {
        b[x] = a[x];
    }
}