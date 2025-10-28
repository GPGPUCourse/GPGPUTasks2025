#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* a,
    __global const uint* pref_bitseqs,
    __global uint* scattered,
    unsigned int n,
    unsigned int bit_offset,
    unsigned int bitseq_len,
    __global const uint* bitseq_totals)
{
    uint i = get_global_id(0);
    if (i >= n) {
        return;
    }
    uint seq = a[i];
    seq >>= bit_offset;
    seq &= ((1u << bitseq_len) - 1);
    uint pos = 0;
    
    pos += bitseq_totals[seq] + pref_bitseqs[(i << bitseq_len) + seq] - 1;
    // printf("i=%d a[i]=%d pos=%d\n", i, a[i],  pos);
    scattered[pos] = a[i];
}