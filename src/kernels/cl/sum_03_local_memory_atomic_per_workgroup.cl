#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);

    if (index >= n / LOAD_K_VALUES_PER_ITEM) {
            return;
    }

    __local unsigned int workgroup_sum;
    if (local_index == 0) {
        workgroup_sum = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint start_addr = index * LOAD_K_VALUES_PER_ITEM;
    uint my_sum = 0;
    for (uint i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
        my_sum += a[start_addr + i];
    }

    atomic_add(&workgroup_sum, my_sum);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0){
        atomic_add(sum, workgroup_sum);
    }

}
