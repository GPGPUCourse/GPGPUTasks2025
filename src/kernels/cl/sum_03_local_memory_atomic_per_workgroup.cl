#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];

    uint my_sum = 0;
    if (index < n / LOAD_K_VALUES_PER_ITEM) {
        for (uint i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
            my_sum += a[i * (n/LOAD_K_VALUES_PER_ITEM) + index];
        }
    }
    local_data[local_index] = my_sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        uint final_sum = 0;
        for (uint i = 0; i < GROUP_SIZE; ++i) {
            final_sum += local_data[i];
        }
        atomic_add(sum, final_sum);
    }

}
