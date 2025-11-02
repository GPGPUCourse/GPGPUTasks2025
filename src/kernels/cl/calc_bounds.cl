#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void calc_bounds(
    __global const uint* sparse_buf,
    __global uint* a_inds,
    __global uint* b_inds,
    int sorted_n
) {
    uint gid = get_global_id(0);
    uint l = gid / (2 * sorted_n) * (2 * sorted_n);
    uint m = l + sorted_n;
    uint t = gid - l;

    uint lhs = -1, rhs = t < sorted_n ? t + 1 : sorted_n;
    // printf("gid=%u l=%u m=%u t=%u rhs=%u\n", gid, l, m, t, rhs);
    while (rhs - lhs > 1) {
        uint mid = (lhs + rhs) / 2;
        if (t - mid >= sorted_n) {
            lhs = mid;
            continue;
        }
        if (sparse_buf[l + mid] > sparse_buf[m + (t - mid)]) {
            rhs = mid;
        } else {
            lhs = mid;
        }
    }
    // printf("gid=%u rhs=%u\n", gid, rhs);
    a_inds[gid] = rhs == 0 ? 0 : rhs - 1;

    lhs = -1; rhs = t < sorted_n ? t + 1 : sorted_n;
    while (rhs - lhs > 1) {
        uint mid = (lhs + rhs) / 2;
        if (t - mid >= sorted_n) {
            lhs = mid;
            continue;
        }
        if (sparse_buf[m + mid] >= sparse_buf[l + (t - mid)]) {
            rhs = mid;
        } else {
            lhs = mid;
        }
    }
    // printf("gid=%u rhs=%u\n", gid, rhs);
    b_inds[gid] = rhs == 0 ? 0 : rhs - 1;
}
