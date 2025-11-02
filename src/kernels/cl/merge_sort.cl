#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

// Partition two sorted ranges along the "merge path" diagonal so individual threads
// can merge disjoint chunks without conflicts.
inline uint2 merge_path_partition(uint diag,
    __global const uint* input_data,
    uint left_start,
    uint left_len,
    uint right_start,
    uint right_len)
{
    uint low = (diag > right_len) ? diag - right_len : 0u;
    if (low > left_len) {
        low = left_len;
    }
    uint high = (diag < left_len) ? diag : left_len;

    while (low < high) {
        uint mid = (low + high) >> 1;
        uint left_idx = mid;
        uint right_idx = diag - mid;

        const int has_left = (int)(left_idx < left_len);
        const int has_right_prev = (int)(right_idx > 0u);

        uint left_val = 0u;
        if (has_left) {
            left_val = input_data[left_start + left_idx];
        }

        uint right_prev = 0u;
        if (has_right_prev) {
            right_prev = input_data[right_start + right_idx - 1u];
        }

        const bool move_low = has_left && has_right_prev && (left_val < right_prev);
        if (move_low) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }

    const uint left_idx = low;
    const uint right_idx = diag - left_idx;
    return (uint2)(left_idx, right_idx);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    if (sorted_k <= 0 || n <= 0) {
        return;
    }

    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint num_groups = get_num_groups(0);

    const ulong block_size64 = (ulong)sorted_k * 2ul;
    const ulong total_segments = block_size64 == 0ul ? 0ul : ((ulong)n + block_size64 - 1ul) / block_size64;

    if (total_segments == 0ul) {
        return;
    }

    for (ulong segment = (ulong)group_id; segment < total_segments; segment += (ulong)num_groups) {
        const ulong block_start64 = block_size64 * segment;
        if (block_start64 >= (ulong)n) {
            continue;
        }

        const uint block_start = (uint)block_start64;
        const ulong left_available = (ulong)n - block_start64;
        const uint left_len = (uint)min(left_available, (ulong)sorted_k);
        const ulong right_start64 = block_start64 + (ulong)left_len;
        const ulong right_available = (right_start64 < (ulong)n) ? ((ulong)n - right_start64) : 0ul;
        const uint right_len = (uint)min(right_available, (ulong)sorted_k);
        const uint total_len = left_len + right_len;
        if (total_len == 0u) {
            continue;
        }

        const uint right_start = (uint)right_start64;
        // Split the merge among all threads in the workgroup.
        const uint chunk = (total_len + GROUP_SIZE - 1u) / GROUP_SIZE;
        const uint begin = min(total_len, local_id * chunk);
        if (begin >= total_len) {
            continue;
        }

        const uint end = min(total_len, begin + chunk);
        const uint out_offset = block_start + begin;
        const uint iterations = end - begin;

        uint2 start_pair = merge_path_partition(begin, input_data, block_start, left_len, right_start, right_len);
        uint left_index = start_pair.x;
        uint right_index = start_pair.y;

        const uint sentinel = 0xffffffffu;
        for (uint k = 0; k < iterations; ++k) {
            const bool has_left = left_index < left_len;
            const bool has_right = right_index < right_len;
            const uint left_value = has_left ? input_data[block_start + left_index] : sentinel;
            const uint right_value = has_right ? input_data[right_start + right_index] : sentinel;

            const bool take_left = has_left && (!has_right || left_value <= right_value);
            output_data[out_offset + k] = take_left ? left_value : right_value;

            left_index += take_left ? 1u : 0u;
            right_index += take_left ? 0u : 1u;
        }
    }
}
