#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    
    if (i >= n) {
        return;
    }
    
    int segment_pair = i / (2 * sorted_k);
    int local_pos = i % (2 * sorted_k);
    
    int seg1_start = segment_pair * 2 * sorted_k;
    int seg1_end = min(seg1_start + sorted_k, n);
    int seg2_start = seg1_end;
    int seg2_end = min(seg2_start + sorted_k, n);
    
    int seg1_size = seg1_end - seg1_start;
    int seg2_size = seg2_end - seg2_start;
    
    int seg1_count;
    int seg2_count;
    
    if (local_pos >= seg1_size + seg2_size) {
        seg1_count = seg1_size;
        seg2_count = seg2_size;
    } else {
        int left = max(0, local_pos - seg2_size);
        int right = min(seg1_size, local_pos + 1);
        seg1_count = left;
        
        while (left < right) {
            int mid = (left + right) >> 1;
            seg2_count = local_pos - mid;
            
            if (seg2_count < 0) {
                right = mid;
                continue;
            }
            
            if (seg2_count > seg2_size) {
                left = mid + 1;
                continue;
            }
            
            bool valid_split = true;
            
            if (mid > 0 && seg2_count < seg2_size) {
                if (input_data[seg1_start + mid - 1] > input_data[seg2_start + seg2_count]) {
                    valid_split = false;
                    right = mid;
                }
            }
            
            if (valid_split && seg2_count > 0 && mid < seg1_size) {
                if (input_data[seg2_start + seg2_count - 1] > input_data[seg1_start + mid]) {
                    valid_split = false;
                    left = mid + 1;
                }
            }
            
            if (valid_split) {
                seg1_count = mid;
                break;
            }
        }
        
        seg2_count = local_pos - seg1_count;
    }
    
    uint result;
    
    if (seg1_count < seg1_size && seg2_count < seg2_size) {
        uint val1 = input_data[seg1_start + seg1_count];
        uint val2 = input_data[seg2_start + seg2_count];
        result = val1 <= val2 ? val1 : val2;
    } else if (seg1_count < seg1_size) {
        result = input_data[seg1_start + seg1_count];
    } else {
        result = input_data[seg2_start + seg2_count];
    }
    
    output_data[i] = result;
}
