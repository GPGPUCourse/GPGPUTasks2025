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
    __global const uint* pref_zeros,
    __global uint* scattered,
    unsigned int n,
    unsigned int bit)
{
    uint i = get_global_id(0);
    if (i >= n) {
        return;
    }
    int pos = (a[i] >> bit & 1) == 0 ? pref_zeros[i] - 1 : pref_zeros[n-1] + i - pref_zeros[i];
    if ((a[i] >> bit & 1) == 0) {
        scattered[pref_zeros[i] - 1] = a[i];
    } else {
        scattered[pref_zeros[n-1] + i - pref_zeros[i]] = a[i];
    }
}