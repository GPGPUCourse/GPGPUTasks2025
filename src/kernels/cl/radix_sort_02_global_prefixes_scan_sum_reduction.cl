#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* buffer1, // входом будет [бин*a2+group]
    __global  uint* buffer2, // выход сумма за бин
    unsigned int a1, // число бинов
    unsigned int a2) // число групп
{
    uint bin = get_global_id(0); // какой бин обрабатывается
    if (bin >= a1)
        return;

    uint sum = 0u;
    for (uint g = 0u; g < a2; ++g)
    {
        sum += buffer1[bin * a2 + g]; // вклад группы добавояю в бин (накоп суммы)
    }
    buffer2[bin] = sum; // итоговая сумма
}
