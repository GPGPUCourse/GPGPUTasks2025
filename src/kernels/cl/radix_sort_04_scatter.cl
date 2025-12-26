#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // ГООООООООООООЙДАААА
    __global const uint* input, // ориг инпут
    __global const uint* pref, // преф суммы по 0 битам в фиксированной позиции бита
    __global       uint* output,
    unsigned int bit,
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;
    if (((input[i] >> bit) & 1u) == 0){
        output[pref[i] - 1] = input[i];
    } else {
        output[pref[n-1] + (i - pref[i])] = input[i];
    }
}