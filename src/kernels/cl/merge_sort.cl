#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const unsigned int* src,
    __global       unsigned int* dst,
             const unsigned int  iter,
             const unsigned int  size
) {
    const unsigned int index = get_global_id(0);

    if (index >= size) {
        return;
    }

    const unsigned int len = 1u << (iter - 1);
    const unsigned int left_bound = (index >> iter) << iter;
    const unsigned int middle = left_bound + len;
    const unsigned int right_bound = middle + len;

    if (middle >= size) {
        dst[index] = src[index];
        return;
    }

    unsigned int left, right, target, const_left, offset;
    if (index < middle) {
        offset = left_bound;
        const_left = left = middle - 1;
        right = min(right_bound, size);
        target = src[index];
    } else {
        offset = middle;
        const_left = left = left_bound - 1;
        right = middle;
        target = src[index] + 1;
    }

    while (right - left > 1) {
        const unsigned int mid = (left + right) / 2;
        if (src[mid] < target) {
            left = mid;
        } else {
            right = mid;
        }
    }

    const unsigned int dst_index = left_bound + (index - offset) + (left - const_left);
    dst[dst_index] = src[index];
}
