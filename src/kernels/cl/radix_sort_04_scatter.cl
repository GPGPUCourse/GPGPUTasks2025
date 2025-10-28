#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* in,
    __global uint* out,
    __global const uint* sum,
    uint bit,
    uint offset,
    uint n)
{
    uint i = get_global_id(0);
    if (i >= n) return;

    uint val = in[i];
    uint is_bit_set = (val & (1 << bit)) != 0;

    uint pos;
    if (!is_bit_set) {
        pos = sum[i] - 1;
    } else {
        pos = (offset - 1) + (i + 1) - sum[i];
    }
    out[pos] = val;

    // if (is_bit_set == consider_bit_set) {
    //     uint pos = sum[i] + offset - 1;
    //     out[pos] = val;
    // }
}