#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"


int binary_search(__global const uint* data,
                  int length,
                  uint target,
                  bool inclusive)
{
    int left = 0;
    int right = length - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        bool move_right = (data[mid] < target) || (inclusive && data[mid] == target);
        
        if (move_right) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return left;
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(__global const uint* input_data,
                         __global uint* output_data,
                         int sorted_k,
                         int n)
{
    const uint global_id = get_global_id(0);
    
    if (global_id >= n) {
        return;
    }
    
    const uint my_group = global_id / sorted_k;
    
    const bool is_even_group = (my_group & 1) == 0;
    const uint neighbor_group = is_even_group ? (my_group + 1) : (my_group - 1);
    
    const uint pos_in_my_group = global_id - my_group * sorted_k;
    
    const uint neighbor_offset = neighbor_group * sorted_k;
    const uint neighbor_length = min(sorted_k, (int)(n - neighbor_offset));
    
    const uint pos_in_neighbor_group = binary_search(
        input_data + neighbor_offset,
        neighbor_length,
        input_data[global_id],
        !is_even_group
    );
    
    const uint merged_group_start = min(my_group, neighbor_group) * sorted_k;
    const uint output_pos = merged_group_start + pos_in_my_group + pos_in_neighbor_group;
    
    if (output_pos < n) {
        output_data[output_pos] = input_data[global_id];
    }
}