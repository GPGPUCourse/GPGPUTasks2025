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
    const unsigned int const_target = src[index];

    if (middle >= size) {
        dst[index] = const_target;
        return;
    }

    unsigned int target = const_target;
    unsigned int left, right, const_left, offset;
    if (index < middle) {
        offset = left_bound;
        const_left = left = middle - 1;
        right = min(middle + len, size);
    } else {
        offset = middle;
        const_left = left = left_bound - 1;
        right = middle;
        target += 1;
    }

    while (right - left > 1) {
        const unsigned int mid = (left + right) / 2;
        const unsigned int cmp = src[mid] < target;
        const unsigned int ncmp = 1 - cmp;
        left = cmp * mid + ncmp * left;
        right = ncmp * mid + cmp * right;
    }

    const unsigned int dst_index = left_bound + (index - offset) + (left - const_left);
    dst[dst_index] = const_target;
}
