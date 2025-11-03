#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#define INF ((uint)0xFFFFFFFFu)

#include "../defines.h"
#include "helpers/rassert.cl"

inline uint2 binsearch(
    __global const uint* A, int a_len,
    __global const uint* B, int b_len,
    int diag)
{
    int total = a_len + b_len;

    if (diag <= 0)
        return (uint2)(0u, 0u);
    if (diag >= total)
        return (uint2)(a_len, b_len);

    int L = max(0, diag - b_len);
    int R = min(diag, a_len);

    while (L <= R) {
        int i = (L + R) >> 1;
        int j = diag - i;

        uint A_im1 = (i > 0) ? A[i - 1] : 0;
        uint B_j = (j < b_len) ? B[j] : INF;
        uint B_jm1 = (j > 0) ? B[j - 1] : 0;
        uint A_i = (i < a_len) ? A[i] : INF;

        if (i > 0 && j < b_len && A_im1 > B_j) {
            R = i - 1;
            continue;
        }

        if (j > 0 && i < a_len && B_jm1 > A_i) {
            L = i + 1;
            continue;
        }
        return (uint2)(i, j);
    }

    int i = L;
    int j = diag - i;
    return (uint2)(i, j);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);

    const int blk_len = sorted_k * 2;
    int blk_start = gid * blk_len;
    if (blk_start >= n)
        return;

    int blk_m = min(blk_start + sorted_k, n);

    int blk_e = min(blk_start + blk_len, n);

    int a_len = blk_m - blk_start;
    int b_len = blk_e - blk_m;
    long total = a_len + b_len;
    if (total <= 0)
        return;

    __global const uint* A = input_data + blk_start;
    __global const uint* B = input_data + blk_m;
    __global uint* C = output_data + blk_start;

    int d_start = (int)((total * (long)lid) / GROUP_SIZE);
    int d_end = (int)((total * (long)(lid + 1)) / GROUP_SIZE);

    if (d_start >= d_end)
        return;

    uint2 st = binsearch(A, a_len, B, b_len, d_start);
    uint2 end = binsearch(A, a_len, B, b_len, d_end);

    int i = st.x;
    int j = st.y;
    int i_end = end.x;
    int j_end = end.y;

    for (int out = d_start; out < d_end; ++out) {
        uint va = (i < i_end) ? A[i] : INF;
        uint vb = (j < j_end) ? B[j] : INF;

        if (va <= vb) {
            C[out] = va;
            ++i;
        } else {
            C[out] = vb;
            ++j;
        }
    }
}
