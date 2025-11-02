#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort_old(
    __global const uint* input_data,
    __global uint* output_data,
    int already_sorted,
    int n)
{
    const unsigned int i = get_global_id(0);

    if (i >= n)
        return;

    uint my_num = input_data[i];

    int left = i & (~(already_sorted * 2 - 1));
    int right = min(left + 2 * already_sorted, n);
    int mid = min(left + already_sorted, n);

    bool is_from_left = i < mid;

    int s_left = (is_from_left ? mid : left) - 1;
    int s_right = is_from_left ? right : mid;

    while (s_right > s_left + 1) {
        int s_mid = (s_left + s_right) / 2;
        // if (i == 0 && already_sorted == 1) {
        //     printf("s_left = %d, s_right = %d, s_mid = %d\n", s_left, s_right, s_mid);
        // }
        uint v_mid = input_data[s_mid];
        bool is_less = is_from_left ? (my_num < v_mid) : (my_num <= v_mid);

        if (!is_less) {
            s_left = s_mid;
        } else
            s_right = s_mid;
        // if (i == 0 && already_sorted == 1) {
        //     printf("s_left = %d, s_right = %d, s_mid = %d\n", s_left, s_right, s_mid);
        // }
    }

    int idx = (i + s_right - mid);

    output_data[idx] = my_num;

    // if (i == 0 && already_sorted == 1) {
    //     printf("i = %d, v = %d, oi = %d, is_from_left = %d, ast = %d\n", i, my_num, idx, is_from_left, already_sorted);
    // }

    // TODO
}
