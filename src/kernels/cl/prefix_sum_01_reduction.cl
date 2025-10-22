#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum,
    __global       uint* next_pow2_sum,
    unsigned int n)
{
    uint idx = get_global_id(0);
    uint out_n = (n + 1u) >> 1u; // размер выходного буфера  вдвое меньше
    if (idx >= out_n)
        return;
    uint i1 = idx << 1; // индекс первого элемента из пары
    uint s  = pow2_sum[i1]; // берем первое значение
    if (i1 + 1 < n)
        s += pow2_sum[i1 + 1]; // если есть второе  прибавляем его
    next_pow2_sum[idx] = s; // сохраняем сумму пары в  буфер на выход
}
