#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* src,
    unsigned srcoff,
    __global const uint* pairs,
    unsigned pairsoff,
    __global       uint* dst,
    unsigned dstoff,
    unsigned int n)
{
    src += srcoff;
    pairs += pairsoff;
    dst += dstoff;
    unsigned i = get_global_id(0);
    if(2 * i + 1 < n)
        dst[2 * i + 1] = pairs[i]; // верим в силу jump threading
    if(2 * i + 2 < n)
        dst[2 * i + 2] = pairs[i] + src[2 * i + 2];
}
