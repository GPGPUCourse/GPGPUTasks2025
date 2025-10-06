#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

// Fix work-group size to GROUP_SIZE from defines.h
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_02_atomics_load_k(__global const uint* a,
                                    __global       uint* sum,
                                           unsigned int  n)
{
    const uint index = get_global_id(0);

    if (index >= n / LOAD_K_VALUES_PER_ITEM) {
        return;
    }

    uint my_sum = 0;
    for (uint i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
        my_sum += a[i * (n/LOAD_K_VALUES_PER_ITEM) + index];
    }

    atomic_add(sum, my_sum);
}
