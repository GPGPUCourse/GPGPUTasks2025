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

    // TODO
    const uint index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_id = get_group_id(0);

    __local uint local_sum;

    // Инициализация локальной суммы
    if (local_index == 0) {
        local_sum = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Каждый work item добавляет свой элемент в локальную сумму
    if (index < n) {
        atomic_add(&local_sum, a[index]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Только первый поток рабочей группы добавляет в глобальную сумму
    if (local_index == 0) {
        atomic_add(sum, local_sum);
    }
}
