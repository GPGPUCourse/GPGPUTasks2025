#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__kernel void prefix_sum_01_reduction(
    __global uint* a,
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

    // if (thr == 0) {
    //     printf("Cur array : ");
    //     for (unsigned int k = 0; k < GROUP_SIZE; k++)
    //         printf("%ld ", tree[k + GROUP_SIZE]);
    //     printf("\n");
    // }

    // build segment tree
    unsigned int l = GROUP_SIZE, r = 2 * GROUP_SIZE - 1;
    while (true) {
        if (thr >= l && thr <= r)
            tree[thr] += tree[2 * thr] + tree[2 * thr + 1];
        
        barrier(CLK_LOCAL_MEM_FENCE);

        // if (thr == 0) {
        //     for (unsigned int k = l; k <= r; k++)
        //         printf("%u ", tree[k]);
        //     printf("\n");
        // }

        l /= 2;
        r /= 2;

        if (l == r)
            break;
    }

    // build prefix sums
    while (true) {
        if (thr >= l && thr <= r) {
            tree[thr * 2 + 1] = tree[thr] + tree[thr * 2];
            tree[thr * 2]     = tree[thr];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // if (thr == 0) {
        //     printf("%ld %ld : ", l, r);
        //     for (unsigned int k = l; k <= r; k++)
        //         printf("%u ", tree[k]);
        //     printf("\n");
        // }

        l = 2 * l;
        r = 2 * r + 1;

        if (l == GROUP_SIZE)
            break;
    }
    // if (thr == 0) {
    //     printf("%ld %ld : ", l, r);
    //     for (unsigned int k = l; k <= r; k++)
    //         printf("%u ", tree[k]);
    //     printf("\n");
    // }

    for (unsigned int k = 0; k < block_len; k++) {
        unsigned int to = i + k + 1 - block_len;
        if (to < n) {
            // printf("val : %ld -> %ld\n",tree[thr + GROUP_SIZE], to);
            a[to] += tree[thr + GROUP_SIZE];
        }
    }
}
