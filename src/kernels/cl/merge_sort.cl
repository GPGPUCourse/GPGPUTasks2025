#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

unsigned int _min(unsigned int a, unsigned int b)
{
    return (a <= b) ? a : b;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const unsigned int i = get_global_id(0);

    if (i >= n)
        return;

    unsigned int big_block_l = i - (i % (2 * sorted_k));

    bool is_left = 1 - ((i / sorted_k) % 2);

    unsigned int another_block_l = _min(big_block_l + sorted_k * is_left, n);
    unsigned int my_block_l = _min(big_block_l + sorted_k * (1 - is_left), n);

    unsigned int l = -1;
    unsigned int r = _min(n - another_block_l, sorted_k);

    // printf("idx %d, find: [%d, %d)\n", i, l, r);

    unsigned int x = input_data[i];

    while (r - l > 1) {
        unsigned int m = (r + l) / 2;
        unsigned int mx = input_data[another_block_l + m];
        if ((mx < x) || (!is_left && mx == x))
            l = m;
        else
            r = m;
    }

    // printf("idx %d, r: %d\n", i, r);

    // printf("input[%d] = %d = output[%d]\n", i, x, big_block_l + (i - my_block_l) + r);

    output_data[big_block_l + (i - my_block_l) + r] = x;
}
