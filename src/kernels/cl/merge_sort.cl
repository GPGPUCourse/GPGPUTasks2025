#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define ITEMS_PER_THREAD 4
#define TILE_OUT (GROUP_SIZE * ITEMS_PER_THREAD)
#define UINT_SENTINEL 0xffffffffu

inline uint2 partition_by_binary_search(
    uint k,
    uint left_len,
    uint right_len,
    __global const uint* left,
    __global const uint* right)
{
    uint low = (k > right_len) ? (k - right_len) : 0;
    uint high = min(k, left_len);
    while (low < high) {
        uint mid = (low + high) >> 1;
        uint b = k - mid;

        uint right_val = (b < right_len) ? right[b] : UINT_SENTINEL;
        if (mid > 0 && left[mid - 1] > right_val) {
            high = mid;
            continue;
        }

        uint left_val = (mid < left_len) ? left[mid] : UINT_SENTINEL;
        if (b > 0 && right[b - 1] >= left_val) {
            low = mid + 1;
            continue;
        }

        low = mid;
        break;
    }

    uint a = low;
    uint b = k - a;
    return (uint2)(a, b);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const uint lid = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint groups_total = get_num_groups(0);

    const uint pair_size = ((uint)sorted_k) << 1;
    if (pair_size == 0 || n == 0) {
        return;
    }

    const uint tiles_per_pair = (pair_size + TILE_OUT - 1u) / TILE_OUT;
    const uint pair_count = (((uint)n) + pair_size - 1u) / pair_size;
    const uint total_tiles = pair_count * tiles_per_pair;

    for (uint tile = group_id; tile < total_tiles; tile += groups_total) {
        uint pair_id = tile / tiles_per_pair;
        uint tile_in_pair = tile - pair_id * tiles_per_pair;

        uint base = pair_id * pair_size;
        if (base >= (uint)n) {
            continue;
        }

        uint left_len = min((uint)sorted_k, (uint)(n - (int)base));
        uint right_len = 0;
        if (base + (uint)sorted_k < (uint)n) {
            right_len = min((uint)sorted_k, (uint)(n - (int)(base + (uint)sorted_k)));
        }
        uint merged_len = left_len + right_len;
        if (merged_len == 0) {
            continue;
        }

        uint k0 = tile_in_pair * TILE_OUT;
        if (k0 >= merged_len) {
            continue;
        }
        uint k1 = k0 + TILE_OUT;
        if (k1 > merged_len) {
            k1 = merged_len;
        }

        __global const uint* left = input_data + base;
        __global const uint* right = input_data + base + (uint)sorted_k;

        uint start = k0 + lid * ITEMS_PER_THREAD;
        for (uint t = 0; t < ITEMS_PER_THREAD; ++t) {
            uint out_pos = start + t;
            if (out_pos >= k1) {
                break;
            }

            uint2 pp = partition_by_binary_search(out_pos, left_len, right_len, left, right);
            uint a = pp.x;
            uint b = pp.y;

            uint left_val = (a < left_len) ? left[a] : UINT_SENTINEL;
            uint right_val = (b < right_len) ? right[b] : UINT_SENTINEL;
            uint value = left_val;
            if (b < right_len && left_val > right_val) {
                value = right_val;
            }

            output_data[base + out_pos] = value;
        }
    }
}
