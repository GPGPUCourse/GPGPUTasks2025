#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* array,
    __global const uint* prefix_sum,
    __global uint* result,
    unsigned int n,
    unsigned int start)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int l_idx = get_local_id(0);
    const unsigned int gr_idx = get_group_id(0);
    const unsigned int b_idx = (array[idx] >> start) % (1 << SORT_BUCKET_SIZE);
    const unsigned int prefix_sum_index = b_idx * ((n + GROUP_SIZE - 1) / GROUP_SIZE) + gr_idx;
    const unsigned int prev_sum = prefix_sum_index == 0 ? 0 :  prefix_sum[prefix_sum_index - 1];
    __local unsigned int b_pr[(1 << SORT_BUCKET_SIZE)];
    __local unsigned int l_pr[GROUP_SIZE][(1 << SORT_BUCKET_SIZE)];

    for (unsigned int i = 0; i < (1 << SORT_BUCKET_SIZE); ++i) {
        l_pr[l_idx][i] = (i == b_idx);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_idx < ((1 << SORT_BUCKET_SIZE))) {
        for (unsigned int i = 1; i < GROUP_SIZE; ++i) {
            l_pr[i][l_idx] += l_pr[i - 1][l_idx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    const unsigned int index = l_pr[l_idx][b_idx] + prev_sum - 1;

    if (idx < n) {
        // printf("%d %d %d %d\n", index, idx, prev_sum, l_pr[l_idx][b_idx]);
        result[index] = array[idx];
    }

}