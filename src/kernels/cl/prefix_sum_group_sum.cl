#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_group_sum(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* as, // input array
    __global       uint* group_sum, // group_sum[i] = sum[i * GROUP_SIZE, (i+1) * GROUP_SIZE)
    unsigned int n, unsigned int group_sum_offset)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    __local uint local_as[GROUP_SIZE];
    local_as[lid] = 0;
    if (gid < n) {
        local_as[lid] = as[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint k = 16;  // столько элементов просуммируем в конце одним потоком

    for (uint pow2 = 1; (1 << pow2) <= GROUP_SIZE / k; pow2++) {
        uint block_size = 1 << pow2;
        uint pair_sum = 0;
        if (lid < GROUP_SIZE / block_size) {
            pair_sum = local_as[2 * lid] + local_as[2 * lid + 1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < GROUP_SIZE / block_size) {
            local_as[lid] = pair_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        uint acc = 0;
        for (int i = 0; i < k; ++i) {
            acc += local_as[i];
        }
        group_sum[group_sum_offset + get_group_id(0)] = acc;
    }
}
