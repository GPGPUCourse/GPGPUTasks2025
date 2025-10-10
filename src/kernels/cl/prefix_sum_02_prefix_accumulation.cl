#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

// __attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* reduced, 
    volatile __global       uint* pref_sum,
    unsigned int n,
    unsigned int p // 2^p is a segment size
)
{
    unsigned int i = get_global_id(0);
    unsigned ind = i + 1;

    if (i < n)
    {
        // 2^p contributes to i
        if ((ind >> p) & 1) 
        {
            unsigned int segment_ind = (ind >> (p + 1)) << 1;
            pref_sum[i] += reduced[segment_ind];
        }
    }
}
