#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void small_merge_sort(
    __global const unsigned int* global_src,
    __global       unsigned int* global_dst,
             const unsigned int  size
) {
    const unsigned int global_index = get_global_id(0);

    if (global_index >= size) {
        return;
    }

    __local unsigned int buf1[GROUP_SIZE];
    __local unsigned int buf2[GROUP_SIZE];
    __local unsigned int* src = buf1;
    __local unsigned int* dst = buf2;

    const unsigned int index = get_local_id(0);
    src[index] = global_src[global_index];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int iter = 1; iter < PIVOT; ++iter) {
        const unsigned int len = 1u << (iter - 1);
        const unsigned int left_bound = (index >> iter) << iter;
        const unsigned int middle = left_bound + len;
        const unsigned int right_bound = middle + len;

        unsigned int left, right, target, const_left, offset;
        if (index < middle) {
            offset = left_bound;
            const_left = left = middle - 1;
            right = min(right_bound, (unsigned int)GROUP_SIZE);
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
        barrier(CLK_LOCAL_MEM_FENCE);

        {
            __local unsigned int* tmp = dst;
            dst = src;
            src = tmp;
        }
    }

    global_dst[global_index] = src[index];
}
