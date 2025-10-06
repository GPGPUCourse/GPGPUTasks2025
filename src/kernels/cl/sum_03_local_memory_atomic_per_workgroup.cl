#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sum_03_local_memory_atomic_per_workgroup(
    __global const uint* a,
    __global uint* sum,
    const unsigned int n)
{
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);

    __local uint local_data[GROUP_SIZE];

    // Store per-work-item partial sums in local memory
    local_data[local_index] = (index < n) ? a[index] : 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction with halving stride
    for (uint stride = GROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (local_index < stride) {
            local_data[local_index] += local_data[local_index + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Only one atomic add per workgroup
    if (local_index == 0u) {
        atomic_add(sum, local_data[0]);
    }
}
