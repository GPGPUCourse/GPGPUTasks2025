#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define WARP_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sum_04_local_reduction(__global const uint* a,
    __global uint* b,
    unsigned int n)
{
    // Подсказки:
    // const uint index = get_global_id(0);
    // const uint local_index = get_local_id(0);
    // __local uint local_data[GROUP_SIZE];
    // barrier(CLK_LOCAL_MEM_FENCE);

    // b - буфер для каскадноо суммирования

    __local uint local_data[GROUP_SIZE];

    uint global_index = get_global_id(0);
    uint local_index = get_local_id(0);

    local_data[local_index] = (global_index < n) ? a[global_index] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Редукция в local memory
    for (uint strd = GROUP_SIZE / 2; strd > 0; strd >>= 1) {
        if (local_index < strd) {
            local_data[local_index] += local_data[local_index + strd];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Каждый workgroup записывает свой результат
    if (local_index == 0) {
        uint group_id = get_group_id(0);
        b[group_id] = local_data[0];
    }
}