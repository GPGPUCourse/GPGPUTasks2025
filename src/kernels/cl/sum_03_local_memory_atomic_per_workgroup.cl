#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const unsigned int* a,
                                                       __global       unsigned int* sum,
                                                       const unsigned int n)
{
    // Подсказки:
    // const unsigned int index = get_global_id(0);
    // const unsigned int local_index = get_local_id(0);
    // __local unsigned int local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);
    // TODO

    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    __local unsigned int local_data[GROUP_SIZE];

    local_data[local_index] = 0;
    if (index < n) {
        local_data[local_index] = a[index];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        unsigned int s = 0;
        for (unsigned int i = 0; i < GROUP_SIZE; ++i) {
            s += local_data[i];
        }
        atomic_add(sum, s);
    }
}
