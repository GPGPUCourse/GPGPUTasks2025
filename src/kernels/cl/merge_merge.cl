#include "../defines.h"
#include "helpers/rassert.cl"

static inline uint merge_path_search_global(
    __global const uint* A,
    __global const uint* B,
    uint diag, uint A_len, uint B_len)
{
    uint kmin = (diag > B_len) ? (diag - B_len) : 0;
    uint kmax = (diag < A_len) ? diag : A_len;

    while (kmin < kmax) {
        uint kmid = (kmin + kmax) >> 1;

        uint a_val = (kmid < A_len) ? A[kmid] : UINT_MAX;

        int b_rel = (int)diag - (int)kmid - 1;
        bool b_neg = (b_rel < 0);
        uint b_val;
        if (b_neg) {
            b_val = 0;
        } else {
            uint ub = (uint)b_rel;
            b_val = (ub < B_len) ? B[ub] : UINT_MAX;
        }

        if (b_neg || a_val > b_val) {
            kmax = kmid;
        } else {
            kmin = kmid + 1;
        }
    }
    return kmin;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_merge(
    __global const uint* input,
    __global uint* output,
    const uint n,
    const uint block_size)
{
    uint local_index = get_local_id(0);
    uint group_index = get_group_id(0);

    uint merge_block_size = block_size << 1;
    uint groups_per_merge = (merge_block_size + GROUP_SIZE - 1) / GROUP_SIZE;

    uint merge_index = group_index / groups_per_merge;
    uint local_group_index = group_index % groups_per_merge;

    uint A_offset = merge_index * merge_block_size;
    uint B_offset = A_offset + block_size;
    if (A_offset >= n) {
        return;
    }

    uint A_len = (A_offset < n) ? min(block_size, n - A_offset) : 0;
    uint B_len = (B_offset < n) ? min(block_size, n - B_offset) : 0;
    if (A_len == 0) {
        return;
    }

    uint diag_start = local_group_index * GROUP_SIZE;
    uint diag_end = min(diag_start + GROUP_SIZE, A_len + B_len);
    if (diag_start >= A_len + B_len) {
        return;
    }

    uint a_start = merge_path_search_global(input + A_offset, input + B_offset, diag_start, A_len, B_len);
    uint b_start = diag_start - a_start;
    uint a_end = merge_path_search_global(input + A_offset, input + B_offset, diag_end, A_len, B_len);
    uint b_end = diag_end - a_end;

    uint rect_a_len = a_end - a_start;
    uint rect_b_len = b_end - b_start;

    uint local_diag = local_index;
    if (local_diag >= rect_a_len + rect_b_len) {
        return;
    }

    uint k = merge_path_search_global(input + A_offset + a_start, input + B_offset + b_start,
        local_diag, rect_a_len, rect_b_len);
    uint a_pos = a_start + k;
    uint b_pos = b_start + (local_diag - k);

    uint result;
    if (k >= rect_a_len || a_pos >= A_len) {
        result = input[B_offset + b_pos];
    } else if ((local_diag - k) >= rect_b_len || b_pos >= B_len) {
        result = input[A_offset + a_pos];
    } else {
        uint av = input[A_offset + a_pos];
        uint bv = input[B_offset + b_pos];
        result = (av <= bv) ? av : bv;
    }

    uint output_pos = A_offset + diag_start + local_diag;
    if (output_pos < n) {
        output[output_pos] = result;
    }
}