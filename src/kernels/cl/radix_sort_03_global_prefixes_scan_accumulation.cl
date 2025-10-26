#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global const uint* buffer1,
    __global  uint* buffer2,
    unsigned int a1,
    unsigned int a2)
{
    uint tid = get_global_id(0); // глобальный id потока
    if (a2 == 0u)
    { // случай нет групп
        if (tid != 0u)
            return; // остальные потоки ничего не делают
        uint acc = 0u;
        for (uint i = 0u; i < a1; ++i)
        { // идём по бинам
            uint t = buffer1[i]; // берём значение
            buffer2[i] = acc; // пишем  префикс
            acc += t; // обновляем сумму
        }
        return;
    }
    // a2 != 0 : паралл по бинам — 1 поток == 1 bin
    uint bin = tid; // номер бина равен id потока
    if (bin >= a1)
        return;

    uint acc = 0u;
    uint base = bin * a2; // смещение начала группы для бина
    for (uint g = 0u; g < a2; ++g)
    {
        uint idx = base + g; // индексм массиве
        uint val = buffer1[idx]; // текущее значение
        buffer2[idx] = acc; // префикс для этой группы
        acc += val;
    }
}

