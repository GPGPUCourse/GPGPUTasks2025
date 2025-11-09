#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"


int count_less(__global const uint* a, uint key, int n) {
    int l = 0;
    int r = n - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (a[m] < key) {
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    return l;
}

int count_less_or_equal(__global const uint* a, uint key, int n) {
    int l = 0;
    int r = n - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (a[m] <= key) {
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    return l;
}

uint find_pos_for_elem(__global const uint* input, uint idx, uint bucket_size, int n) {
    uint bucket_id = idx / bucket_size;
    uint id_in_bucket = idx % bucket_size;

    bool is_left_bucket = ((bucket_id & 1) == 0);

    uint other_bucket_id = is_left_bucket ? (bucket_id + 1) : (bucket_id - 1);
    uint bucket_start = bucket_id * bucket_size;
    uint other_bucket_start = other_bucket_id * bucket_size;

    if (other_bucket_start >= (uint)n) {
        return idx;
    }

    int size_in_other = max(0, min((int)bucket_size, n - (int)other_bucket_start));
    uint value = input[idx];

    if (is_left_bucket) {
        int less = count_less(input + other_bucket_start, value, size_in_other);
        return bucket_start + id_in_bucket + (uint)less;
    } else {
        int less_or_eq = count_less_or_equal(input + other_bucket_start, value, size_in_other);
        return other_bucket_start + id_in_bucket + (uint)less_or_eq;
    }

}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  bucket_size,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if (i >= (uint)n) {
        return;
    }

    uint pos = find_pos_for_elem(input_data, i, bucket_size, n);
    output_data[pos] = input_data[i];
}
