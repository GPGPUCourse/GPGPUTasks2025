#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

int lower_bound(__global const uint* data, uint value, int n) {
    int l = 0, r = n - 1;
    while (l <= r) {
        int mid = (l + r) / 2;
        if (data[mid] < value) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return l;
}

int upper_bound(__global const uint* data, uint value, int n) {
    int l = 0, r = n - 1;
    while (l <= r) {
        int mid = (l + r) / 2;
        if (data[mid] <= value) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return l;
}

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

    uint bucket_size = sorted_k;
    uint bucket = i / bucket_size;
    uint another_bucket = (bucket % 2 == 0) ? (bucket + 1) : (bucket - 1);

    uint bucket_start = bucket * bucket_size;
    uint another_bucket_start = another_bucket * bucket_size;
    if (another_bucket_start >= n) {
        output_data[i] = input_data[i];
        return;
    }

    if (another_bucket < bucket) {
        int index = upper_bound(input_data + another_bucket_start, input_data[i], bucket_size);
        output_data[another_bucket_start + i % bucket_size + index] = input_data[i];
    } else {
        int index = lower_bound(input_data + another_bucket_start, input_data[i], min(bucket_size, n - another_bucket_start));
        output_data[bucket_start + i % bucket_size + index] = input_data[i];
    }
}