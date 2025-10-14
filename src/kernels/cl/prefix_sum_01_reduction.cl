#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__kernel void prefix_sum_01_reduction(
    __global uint* a,
    __global uint* pref_sparse,
    unsigned int n, 
    unsigned int block_len)
{
    unsigned int thr = get_local_id(0);
    unsigned int i = (get_global_id(0) + 1) * block_len - 1;
    __local unsigned int tree[2 * GROUP_SIZE];
    
    if (i < n)
        tree[thr + GROUP_SIZE] = a[i];
    else
        tree[thr + GROUP_SIZE] = 0;

    tree[thr] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // build segment tree
    for (unsigned int l = GROUP_SIZE, r = 2 * GROUP_SIZE - 1; l != r; l /= 2, r /= 2) {
        if (thr >= l && thr <= r)
            tree[thr] += tree[2 * thr] + tree[2 * thr + 1];
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // build prefix sums
    for (unsigned int l = 1, r = 1; l != GROUP_SIZE; l <<= 1, r = (r << 1) + 1) {
        if (thr >= l && thr <= r) {
            tree[thr * 2 + 1] = tree[thr] + tree[thr * 2];
            tree[thr * 2]     = tree[thr];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int k = 0; k < block_len; k++) {
        unsigned int to = i + k + 1 - block_len;
        if (to < n) {
            // printf("val : %ld -> %ld\n",tree[thr + GROUP_SIZE], to);
            a[to] += tree[thr + GROUP_SIZE];
        }
    }
}
