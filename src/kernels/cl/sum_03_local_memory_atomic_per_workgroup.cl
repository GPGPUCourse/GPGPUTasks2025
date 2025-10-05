#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{

    const uint index = get_local_id(0);
    const uint local_index = get_global_id(0);
    
    __local uint shared_mem[GROUP_SIZE];

    uint value_to_load = (local_index < n) ? a[local_index] : 0;
    shared_mem[index] = value_to_load;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (index == 0) {
        uint partial_result = 0;
        for (size_t offset = 0; offset < GROUP_SIZE; offset++) {
            partial_result += shared_mem[offset];
        }
        atomic_add(sum, partial_result);
    }
}
