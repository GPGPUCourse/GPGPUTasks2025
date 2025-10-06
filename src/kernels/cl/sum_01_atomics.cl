#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_01_atomics(__global const uint* a,
                             __global       uint* sum,
                                    unsigned int  n)
{
    const uint index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(sum, a[index]);
}
