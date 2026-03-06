#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
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
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const int my_block_num = i / sorted_k; 
    const int friend_block_num = (my_block_num ^ 1);
    const bool is_friend_to_the_right = (my_block_num < friend_block_num);
    int l = friend_block_num * sorted_k - 1;
    int r = min((friend_block_num + 1) * sorted_k, n);
    if (l >= n - 1) {
        output_data[i] = input_data[i];
        return;
    }
    while ((r - l) > 1) {
        int m = l + (r - l) / 2;
        bool friend_goes_before_me = is_friend_to_the_right
            ? input_data[i] > input_data[m]
            : input_data[m] <= input_data[i];
        if (friend_goes_before_me) {
            l = m;
        } else {
            r = m;
        }
    }
    output_data[min(my_block_num, friend_block_num) * sorted_k + (i - my_block_num * sorted_k) + (r - friend_block_num * sorted_k)] = input_data[i];
}
