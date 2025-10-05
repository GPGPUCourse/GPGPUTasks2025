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
    const uint size = get_global_size(0);
    const uint local_size = get_local_size(0);

    __local uint local_data[GROUP_SIZE];

    uint part_sum = 0;
    for (uint idx = index; idx < n; idx += size) {
        part_sum += a[idx];
    }
    local_data[local_index] = part_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // я здесь попробовала сделать так, чтобы на каждом шаге потоки работали по парам и на каждом шаге сокращаем количество работающих потоков
    // у меня это получилось эффективнее, чем 1 мастер тред все складывает, но здесь барьер дорогой
    for (uint step = local_size / 2; step > 0; step = step / 2) {
        if (local_index < step) {
            local_data[local_index] += local_data[local_index + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) { // мастер-поток
        atom_add(sum, local_data[0]);
    }
}
