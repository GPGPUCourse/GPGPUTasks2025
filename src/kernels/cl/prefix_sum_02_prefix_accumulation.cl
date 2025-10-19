#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    unsigned int n,
    unsigned int stride)
{
    const unsigned int index = get_global_id(0);

    if (index > (n + (1 << stride) - 2) / (1 << stride))
        return;

    int index_sum = n - 1 - index * (1 << stride);

    int index_sum_prev = n - 1 - (index + 1) * (1 << (stride));

    if (index_sum < n && index_sum >= 0) {
        if (index_sum - (1 << (stride - 1)) >= 0 && index_sum_prev >= 0)
            pow2_sum[index_sum - (1 << (stride - 1))] += pow2_sum[index_sum_prev];
        // else + 0
    }
}
