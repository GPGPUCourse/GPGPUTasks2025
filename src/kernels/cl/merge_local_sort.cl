#include "../defines.h"
#include "helpers/rassert.cl"

static inline uint merge_path_search_local(
    __local const uint* A,
    __local const uint* B,
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
merge_local_sort(
    __global uint* data,
    const uint n)
{
    uint global_index = get_global_id(0);
    uint local_index = get_local_id(0);
    uint group_id = get_group_id(0);
    uint group_size = get_local_size(0);

    uint group_offset = group_id * group_size;

    __local uint local_data[GROUP_SIZE << 1];

    if (global_index < n) {
        local_data[local_index] = data[global_index];
    } else {
        local_data[local_index] = UINT_MAX;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint sub_block_size = 1; sub_block_size < group_size; sub_block_size <<= 1) {
        uint block_size = sub_block_size << 1;
        uint block_id = local_index / block_size;
        uint block_start = block_id * block_size;

        uint left_start = block_start;
        uint left_end = min(left_start + sub_block_size, group_size);
        uint right_start = left_end;
        uint right_end = min(right_start + sub_block_size, group_size);

        uint diag = local_index - block_start;
        uint left_len = left_end - left_start;
        uint right_len = right_end - right_start;

        if (local_index >= left_start && local_index < right_end) {
            uint k = merge_path_search_local(
                local_data + left_start,
                local_data + right_start,
                diag, left_len, right_len);

            uint li = k;
            uint ri = diag - k;

            uint result;
            if (li >= left_len) {
                result = local_data[right_start + ri];
            } else if (ri >= right_len) {
                result = local_data[left_start + li];
            } else {
                uint lv = local_data[left_start + li];
                uint rv = local_data[right_start + ri];
                result = (lv <= rv) ? lv : rv;
            }
            local_data[group_size + local_index] = result;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_index >= left_start && local_index < right_end) {
            local_data[local_index] = local_data[group_size + local_index];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_index < n) {
        data[global_index] = local_data[local_index];
    }
}
