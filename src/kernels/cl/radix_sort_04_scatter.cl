#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* prefix_sum_accum,
    __global const uint* map_gpu_buffer,
    __global const uint* input_gpu_copy,
    __global uint* buffer_output_gpu,
    unsigned int n)
{
    size_t gid = get_global_id(0);
    if (gid >= n) return;
    uint local_prefix = prefix_sum_accum[gid];
    uint ones = prefix_sum_accum[n - 1] + map_gpu_buffer[n - 1];

    if (map_gpu_buffer[gid]) {
        uint pos = n - ones + local_prefix;
        buffer_output_gpu[pos] = input_gpu_copy[gid];
    } else {
        uint pos = gid - local_prefix;
        buffer_output_gpu[pos] = input_gpu_copy[gid];
    }
}