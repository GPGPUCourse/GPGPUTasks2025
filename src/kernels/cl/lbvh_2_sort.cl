#include "../defines.h"

__kernel void lbvh_2_sort(
    __global const uint* in_keys,
    __global const uint* in_values,
    __global       uint* out_keys,
    __global       uint* out_values,
                   int  sorted_k,
                   int  n)
{
    int global_id = get_global_id(0);
    if (global_id >= n) return;

    int chunk_idx = global_id / sorted_k;
    int chunk_offset = global_id % sorted_k;

    bool is_left = (chunk_idx % 2 == 0);
    int my_start = chunk_idx * sorted_k;
    int other_start = is_left ? (chunk_idx + 1) * sorted_k : (chunk_idx - 1) * sorted_k;

    if (other_start >= n) {
        out_keys[global_id] = in_keys[global_id];
        out_values[global_id] = in_values[global_id];
        return;
    }
    
    int other_len = min(sorted_k, n - other_start);

    uint my_key = in_keys[global_id];

    int low = 0;
    int high = other_len;

    while (low < high) {
        int mid = (low + high) / 2;
        uint other_key = in_keys[other_start + mid];
        
        bool move_right;
        if (is_left) {
             move_right = (other_key < my_key);
        } else {
             move_right = (other_key <= my_key);
        }
        
        if (move_right) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    
    int elements_from_other_before_me = low;

    int merged_start = min(my_start, other_start);
    int final_pos = merged_start + chunk_offset + elements_from_other_before_me;
    
    out_keys[final_pos] = my_key;
    out_values[final_pos] = in_values[global_id];
}

