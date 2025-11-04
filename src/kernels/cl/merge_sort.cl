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
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    const unsigned int sorted_2k = 2 * sorted_k;

    const unsigned int pair_id = i / sorted_2k;
    unsigned L = pair_id * sorted_2k;
    unsigned M = L + sorted_k;
    if (M > n) M = n;
    unsigned R = L + sorted_2k;
    if (R > n) R = n;

    unsigned len1 = M - L;
    unsigned len2 = R - M;
    if (i < M) {
        unsigned local_i = i - L;
        unsigned val = input_data[L + local_i];

        unsigned l = 0;
        unsigned r = len2;

        while (l < r) {
            unsigned mid = (l + r) >> 1;
            unsigned x = input_data[mid + M];
            if (x <= val) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        unsigned out_pos = L + local_i + l;
        if (out_pos < n) output_data[out_pos] = val;
    } else {
        if (i >= R) return;
        unsigned local_i = i - M;
        unsigned val = input_data[M + local_i];

        unsigned l = 0;
        unsigned r = len1;

        while (l < r) {
            unsigned mid = (l + r) >> 1;
            unsigned x = input_data[L + mid];
            if (x < val) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        unsigned out_pos = L + local_i + l;
        if (out_pos < n) output_data[out_pos] = val;
    }
}
