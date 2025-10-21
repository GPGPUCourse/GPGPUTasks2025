#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    // __global const uint* pow2_sum, // contains n values
    // __global       uint* next_pow2_sum, // will contain (n+1)/2 values
    __global uint* buf, // will contain (n+1)/2 values
    uint start,
    uint end,
    __global const uint* inp // will contain (n+1)/2 values
    // unsigned int n
     )
{
    uint i = get_global_id(0);
    uint l = get_local_id(0);
    bool active = start + i < end;

    __local uint prefixes[256];
    {
        uint val = 0;
        if (active){
            if (start != 0) val = buf[start + i];
            else val = inp[start + i];}
        prefixes[l] = val;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    uint delta = 1;
    while (delta < 256) {
        if ((l & (delta * 2 - 1)) == (delta * 2 - 1))
            prefixes[l] += prefixes[l - delta];
        barrier(CLK_LOCAL_MEM_FENCE);
        delta *= 2;
    }

    uint res = 0;

    uint sh = 0;
    const uint sl = l + 1;
    while (sh < 9) {
        const uint shifted = 1 << sh;
    if (sl & shifted) res += prefixes [(sl & (~(shifted - 1))) - 1];

    sh += 1;
    }

    if (active) buf [start + i] = res;

    if (l == 255) {
        buf [end + i / 256] = res;
        // printf ("address: %u, end = %u, i = %u\n",end + i / 256, end, i);
    }

    // printf ("l: %u, res: %u\n", l, res);


    // TODO
}
