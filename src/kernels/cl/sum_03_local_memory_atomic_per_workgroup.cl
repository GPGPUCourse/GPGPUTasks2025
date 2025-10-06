#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
    __global uint* sum,
    const unsigned int n)
{
    // Подсказки:
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];
    if (index >= n)
        local_data[local_index] = 0;
    else
        local_data[local_index] = a[index];
    uint allowed = GROUP_SIZE / 2;
    while (allowed) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if ((allowed > local_index)) {
            local_data[local_index] += local_data[local_index + allowed];
        }
        allowed /= 2;
    }
    if (local_index == 0) {
        // uint res = 0;
        // for (uint i = 0; i < GROUP_SIZE; i++) {
        //     res += local_data[i];
        // }
        // printf("groupid = %u %u\n", get_group_id(1), get_group_id(2));
        atomic_add(sum, local_data[0]);
    }

    // TODO
}
