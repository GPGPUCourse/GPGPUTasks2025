#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__kernel void prefix_sum_01_reduction(
    __global uint* block_sums_in,
    __global uint* block_sums_out,
    __global uint* buffer,
    unsigned int n,
    unsigned int block_len)
{
    unsigned int thr = get_local_id(0);
    unsigned int i = get_global_id(0);
    unsigned int group = get_group_id(0);
    unsigned int block_num = (n + block_len - 1) / block_len;
    __local unsigned int tree[2 * GROUP_SIZE];

    if (i < block_num)
        tree[thr + GROUP_SIZE] = block_sums_in[i];
    else
        tree[thr + GROUP_SIZE] = 0;

    tree[thr] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // if (thr == 0) {
    //     printf("group : %ld\n", group);
    // }
    // build segment tree
    for (unsigned int l = GROUP_SIZE, r = (2 * GROUP_SIZE - 1); l != 0; l /= 2, r /= 2) {
        if (thr >= l && thr <= r)
            tree[thr] = tree[2 * thr] + tree[2 * thr + 1];
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thr == 0) {
        // printf("group id : %ld sum = %ld\n", group, tree[1]);
        block_sums_out[group] = tree[1];
        tree[1] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // if (thr == 0) {
    //     printf("segment tree\n");
    // }
    // for (unsigned int l = GROUP_SIZE, r = (2 * GROUP_SIZE - 1); l != 0; l /= 2, r /= 2) {
    //     if (thr == 0) {
    //         for (unsigned int k = l; k <= r; k++)
    //             printf("%u ", tree[k]);
    //         printf("\n");
    //     }
    // }

    // build prefix sums
    for (unsigned int l = 1, r = 1; l != GROUP_SIZE; l <<= 1, r = (r << 1) + 1) {
        if (thr >= l && thr <= r) {
            tree[thr * 2 + 1] = tree[thr] + tree[thr * 2];
            tree[thr * 2] = tree[thr];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // if (thr == 0) {
    // printf("prefix tree\n");
    // }
    // for (unsigned int l = GROUP_SIZE, r = (2 * GROUP_SIZE - 1); l != 0; l /= 2, r /= 2) {
    //     if (thr == 0) {
    //         for (unsigned int k = l; k <= r; k++)
    //             printf("%u ", tree[k]);
    //         printf("\n");
    //     }
    // }

    // for (unsigned int k = 0; k < block_len; k++) {
    //     unsigned int ind = (get_global_id(0) + 1) * block_len - 1;
    //     unsigned int to = ind + k + 1 - block_len;
    //     if (to < n) {
    //         // printf("val : %ld -> %ld\n",tree[thr + GROUP_SIZE], to);
    unsigned int to = group * GROUP_SIZE + thr;
    if (to < n)
        buffer[group * GROUP_SIZE + thr] = tree[thr + GROUP_SIZE];
    //     }
    // }
}
