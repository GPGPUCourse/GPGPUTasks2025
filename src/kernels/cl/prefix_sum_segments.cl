#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel void prefix_sum_segments(
    __global const uint* a, // source numbers or from previous iteration
    __global       uint* b, // results of current iteration
    unsigned int k, // current length of computed sums (pow of 2)
    unsigned int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const unsigned int segment_start_mask = ~(k / 2 - 1);
    const int prev_segment_end = (i & segment_start_mask) - 1;
    const uint prev_segment_end_sum = prev_segment_end < 0 ? 0 : a[prev_segment_end];
    if (i % k < k / 2) {
        b[i] = a[i];
    } else {
        b[i] = a[i] + prev_segment_end_sum;
    }
    // printf("ocl iteration: %d of %d with pow %d load %d store %d\n", i, n, k, a[i], b[i]);
}
