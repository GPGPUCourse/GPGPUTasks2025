#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* buf, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    unsigned int n,
    __global       uint* out // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    )
{
    // TODO


    uint i = get_global_id (0);

    if (i >= n) return;
    
    uint offset = 0;
    uint tn = n;
    uint ti = i + 1;
    uint res = 0;

    while (tn > 1) {
        if ((ti & 255) != 0)
            // res += buf [offset + (ti & 255) - 1];
            res += buf [offset + (ti) - 1];


        offset += tn;
        tn -= 1;
        tn /= 256;
        tn += 1;

        ti /= 256;
    }

    out [i] = res;
    
}
