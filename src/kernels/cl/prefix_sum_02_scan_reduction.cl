#include "helpers/rassert.cl"
#include "../defines.h"

// reduce блоков которые собрали в первом кернеле
__kernel void prefix_sum_02_scan_reduction(
    __global int* input,
    __global int* output,
    size_t n_blocks,
    size_t stride)
{
    size_t gid = get_global_id(0);

    if (gid < n_blocks) {
        if (gid >= stride) {
            output[gid] = input[gid] + input[gid - stride];
        } else {
            output[gid] = input[gid];
        }
    }
}

