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
     const uint index = get_global_id(0);
     const uint local_index = get_local_id(0);
     __local uint local_data[GROUP_SIZE];
//     barrier(CLK_LOCAL_MEM_FENCE);
    uint value = (index < n) ? a[index] : 0; // чтобы не выйти за границы
    local_data[local_index] = value; // загрузка в локал память
    barrier(CLK_LOCAL_MEM_FENCE); // ожидание пока все загрузится

    // вычисление локальной суммы в группе
    for (uint offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
        if (local_index < offset)
            local_data[local_index] += local_data[local_index + offset];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) // первый в группе добавляется в глоб сумму
        atomic_add(sum, local_data[0]);
}
