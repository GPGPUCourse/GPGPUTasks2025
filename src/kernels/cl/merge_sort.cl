#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    int segment = i / (2 * sorted_k);
    int local_i = i % (2 * sorted_k);

    int left_start = segment * 2 * sorted_k;
    int left_end = min(left_start + sorted_k, n);
    int right_start = left_end;
    int right_end = min(right_start + sorted_k, n);

    int left_size = left_end - left_start;
    int right_size = right_end - right_start;

    if (local_i < left_size + right_size) {
        int pos_in_left = 0;
        int pos_in_right = 0;

        if (local_i < left_size) {
            uint val = input_data[left_start + local_i];

            int l = 0, r = right_size;
            while (l < r) {
                int mid = (l + r) / 2;
                if (input_data[right_start + mid] < val) {
                    l = mid + 1;
                } else {
                    r = mid;
                }
            }
            pos_in_right = l;
            pos_in_left = local_i;
        } else {
            uint val = input_data[right_start + (local_i - left_size)];

            int l = 0, r = left_size;
            while (l < r) {
                int mid = (l + r) / 2;
                if (input_data[left_start + mid] <= val) {
                    l = mid + 1;
                } else {
                    r = mid;
                }
            }
            pos_in_left = l;
            pos_in_right = local_i - left_size;
        }

        output_data[left_start + pos_in_left + pos_in_right] =
            (local_i < left_size) ? input_data[left_start + local_i] : input_data[right_start + (local_i - left_size)];
    }
}
