#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#define MAX_LSZ 256
#define MAX_BINS 2048
__attribute__((reqd_work_group_size(MAX_LSZ, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1, // вход
    __global const uint* buffer2, // префикс по бинам
    __global  uint* buffer3, // выход по тек разряду
    __global const uint* buffer4, // локальные оффсеты по бинам
    unsigned int a1, // сдвиг разряда
    unsigned int a2, // число бин
    unsigned int a3) // кол-во эл
{
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    const uint gid = get_global_id(0);
    const uint grp = get_group_id(0);

    __local uint lstart[MAX_BINS]; // локальный старт для каждой корзины
    __local uint lbin  [MAX_LSZ]; // локальные номера корзин для потоков
    __local uint lact  [MAX_LSZ]; // флаги активности потоков

    uint num_groups = get_num_groups(0) * get_num_groups(1); // колв-о групп (X*Y)
    for (uint i = lid; i < a2; i += lsz)
    { // инит локальные старты по бинам до конца
        uint off = buffer4[i * num_groups + grp]; // вклад текущей группы для бина i
        lstart[i] = buffer2[i] + off; // общий старт бина + локальный оффсет группы
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint v = 0u; // значение дефолт
    uint b = 0u; // бин дефолт
    uint active = (gid < a3) ? 1u : 0u; // проверка есть ли элемент
    if (active)
    {
        v = buffer1[gid]; // читаем значение
        b = (v >> a1) & (a2 - 1u); // вычисляем корзину по разряду
    }

    lbin[lid] = b; // номер корзины потока
    lact[lid] = active; // флаг активности
    barrier(CLK_LOCAL_MEM_FENCE);

    if (active)
    {
        uint local_rank = 0u; // локальный ранг внутри корзины
        for (uint t = 0u; t < lid; ++t)
        { // сколько таких же корзин
            uint same = (lact[t] && (lbin[t] == b)) ? 1u : 0u; // 1 если активен и тот же бин
            local_rank += same;
        }
        uint dst = lstart[b] + local_rank; // финал позиция элемента
        buffer3[dst] = v;
    }

}