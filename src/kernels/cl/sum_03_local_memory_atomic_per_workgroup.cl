#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sum_03_local_memory_atomic_per_workgroup(__global const uint* a,
                                                       __global       uint* sum,
                                                       const unsigned int n)
{
    const unsigned int index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);
    __local uint local_data[GROUP_SIZE];

    if (index < n) {
        local_data[local_index] = a[index];
    } else {
        local_data[local_index] = 0;
    }

    // да, то что ниже не очень красиво, но это дало увеличение bandwidth примерно в два раза (с ~55 до >100 Gb/s)
    // цикл не хотел использовать, потому что от него алгоритм упрется больше в alu-сложность, преимущество
    // перед тем что ниже у него только в компактности написания
    if (!(local_index & 7)) {
        local_data[local_index] += local_data[local_index + 1];
        local_data[local_index] += local_data[local_index + 2];
        local_data[local_index] += local_data[local_index + 3];
        local_data[local_index] += local_data[local_index + 4];
        local_data[local_index] += local_data[local_index + 5];
        local_data[local_index] += local_data[local_index + 6];
        local_data[local_index] += local_data[local_index + 7];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        unsigned int group_sum = 0;
        for (uint i = 0; i < GROUP_SIZE; i += 8) {
            group_sum += local_data[i];
        }
        atomic_add(sum, group_sum);
    }
}
