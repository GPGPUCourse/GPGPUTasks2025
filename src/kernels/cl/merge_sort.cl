#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const uint i = get_global_id(0);
    if (i >= n) {
        return;
    }

    const uint bucket_size = 2 * sorted_k;
    const uint bucket_idx = i / bucket_size;
    const int bucket_start = bucket_idx * bucket_size;

    __global const uint* a = input_data + bucket_start;
    __global const uint* b = input_data + bucket_start + sorted_k;

    uint index_in_bucket = i - bucket_start;
    __global const uint* to_compare = NULL;

    int l = -1;
    int r;
    if (index_in_bucket >= sorted_k) {
        to_compare = a;
        r = sorted_k;
    } else {
        to_compare = b;
        r = max(0, min(sorted_k, n - (bucket_start + sorted_k)));
    }

    while (l < r - 1) {
        int m = l + (r - l) / 2;

        bool cond;
        if (index_in_bucket >= sorted_k) {
            cond = to_compare[m] < input_data[i];
        } else {
            cond = to_compare[m] <= input_data[i];
        }

        if (cond) {
            l = m;
        } else {
            r = m;
        }
    }
    int idx = r;

    if (index_in_bucket >= sorted_k) {
        index_in_bucket -= sorted_k;
    }

    output_data[bucket_start + index_in_bucket + idx] = input_data[i];
}
