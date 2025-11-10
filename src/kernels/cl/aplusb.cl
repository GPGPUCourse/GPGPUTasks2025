#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void aplusb(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                            unsigned int  n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    if (index == 0) {
        printf("OpenCL printf test in aplusb.cl kernel! a[index]=%d b[index]=%d \n", a[index], b[index]);
    }

    rassert(a[index] == 3 * (index + 5) + 7, 43562543223);
    rassert(b[index] == 11 * (index + 13) + 17, 546365435);

    c[index] = a[index] + b[index];
}
