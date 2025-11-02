#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
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

    const unsigned int tmp = src[index];
    //printf("iter:%u index:%u left:%u right:%u target:%u src:%u const_left:%u offset:%u\n", iter, index, left, right, target, tmp, const_left, offset);

    while (right - left > 1) {
        const unsigned int mid = (left + right) / 2;
        if (src[mid] < target) {
            left = mid;
        } else {
            right = mid;
        }
    }

    const unsigned int dst_index = left_bound + (index - offset) + (left - const_left);
    //printf("iter:%u index:%u left_bound:%u offset:%u left:%u const_left:%u dst_index:%u\n", iter, index, left_bound, offset, left, const_left, dst_index);
    dst[dst_index] = src[index];
}

/*
    if (index > (lsz + rsz - 1)) {
        return;
    }

    __global const unsigned int* rhs = lhs + off;

    unsigned int lhs_right = min(index + 1, lsz);
    unsigned int lhs_left;
    if (lhs_right < rsz) {
        lhs_left = -1;
    } else {
        lhs_left = lhs_right - rsz;
    }

    while (lhs_right - lhs_left > 1) {
        const unsigned int lhs_mid = (lhs_left + lhs_right) / 2;
        const unsigned int rhs_mid = index - lhs_mid;
        if (lhs[lhs_mid] <= rhs[rhs_mid]) {
            lhs_left = lhs_mid;
        } else {
            lhs_right = lhs_mid;
        }
    }

    const unsigned int lhs_index = lhs_right;
    const unsigned int rhs_index = index - lhs_index;
    const unsigned int target_index = lhs_index + rhs_index;

    unsigned int target;
    if (lhs_index == lsz || lhs[lhs_index] > rhs[rhs_index]) {
        target = rhs[rhs_index];
    } else {
        target = lhs[lhs_index];
    }
    res[target_index] = target;
*/
