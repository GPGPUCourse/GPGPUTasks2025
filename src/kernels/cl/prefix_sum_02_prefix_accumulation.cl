#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n,
    unsigned int pow2)
{
    unsigned int id = get_global_id(0);

    if (id < n) {
        unsigned int orig_id = (id + 1) / pow2 - 1;

        if (!(orig_id & 1)) {
            prefix_sum_accum[id] += pow2_sum[orig_id];
        }
    }
}