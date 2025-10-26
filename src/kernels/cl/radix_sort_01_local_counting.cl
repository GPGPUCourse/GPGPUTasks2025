#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#define MAX_BINS 2048

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buffer1, // массив входно
    __global uint* buffer2, // выходной
    unsigned int a1, // какой разряд сортируется
    unsigned int a2, // число корзин
    unsigned int a3) // кол-во элемнто во входе
{
    const uint lid = get_local_id(0);
    const uint lsz = get_local_size(0);
    const uint gid = get_global_id(0);
    const uint gsz = get_global_size(0);
    const uint grp = get_group_id(0);

    __local uint lhist[MAX_BINS]; // local гистограмма по горзинам
    for (uint i = lid; i < a2; i += lsz) // гистограмму иниц нулями по порядку
        lhist[i] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint idx = gid; idx < a3; idx += gsz)
    { // читает значение из буффера и вычисляет номер корзины по битам а затем увеличивате счетки в в памяти
        uint v = buffer1[idx];
        uint bin = (v >> a1) & (a2 - 1u);
        atomic_inc((volatile __local uint*)&lhist[bin]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint num_groups = get_num_groups(0) * get_num_groups(1); // число групп по X и Y (если Y=1, то просто по X)
    for (uint i = lid; i < a2; i += lsz) {
        uint out = i * num_groups + grp;
        buffer2[out] = lhist[i];
    }
}
