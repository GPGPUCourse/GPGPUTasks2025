#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void merge_sort(
    __global const uint* a,
    __global       uint* sorted_segments,
                   int  sorted_block_len,
                   int  n)
{
    unsigned int i = get_global_id(0);
    unsigned int thr = get_local_id(0);
    unsigned int block_id = i / sorted_block_len;

    if (i < n) {
        unsigned int key = a[i];
        unsigned int block_offset = (block_id + ((block_id & 1) ? -1 : 0)) * sorted_block_len;
        unsigned int corresp_block_id = block_id + ((block_id & 1) ? -1 : +1);
        int l = -1 + corresp_block_id * sorted_block_len;
        int r = sorted_block_len + corresp_block_id * sorted_block_len;
        // printf("i = %ld, l = %ld, r = %ld\n", i, l, r);
        // inv a[r] is the leftmost bigger than a[i] for left segmenmt
        //     a[r] is the leftmost bigger or equal
        while (l < r - 1) {
            unsigned int m = (l + r) / 2;
            if (m >= n || a[m] + ((block_id & 1) ? +1 : 0) > key)
                r = m;
            else
                l = m; 
        }
        // printf("r = %ld\n", r);

        unsigned int rel_pos_of_r = r - corresp_block_id * sorted_block_len;
        unsigned int rel_pos_of_i = i % sorted_block_len;
        unsigned int to = rel_pos_of_i + rel_pos_of_r + block_offset;
        // printf("%ld -> %ld, r = %ld, rel_pos - %ld, block_offset = %ld\n", i, to, r, rel_pos_of_i, block_offset);
            
        if (to < n) {
            sorted_segments[to] = a[i];
        }
    }
}
