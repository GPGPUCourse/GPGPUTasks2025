#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

static inline uint get_elem(__global const uint* data, int base, int data_len, int i) {
    if (i < 0) {
        return 0u;
    }
    if (i >= data_len) {
        return 0xFFFFFFFFu;
    }

    return data[base + i];
}

static inline int upper_bound(__global const uint* data, int base, int data_len, uint val) {
    int left = 0;
    int right = data_len;
    while (left < right) {
        int m = (left + right) / 2;
        if (data[base + m] <= val) {
            left = m + 1;
        } else {
            right = m;
        }
    }
    return left;
}

static inline int lower_bound(__global const uint* data, int base, int data_len, uint val) {
    int left = 0;
    int right = data_len;
    while (left < right) {
        int m = (left + right) / 2;
        if (data[base + m] < val) {
            left = m + 1;
        } else {
            right = m;
        }
    }
    return left;
}


__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = get_global_id(0);
    if ((int)i >= n) {
        return;
    }
    const int block_sz = 2 * sorted_k;
    const int block_id = i / block_sz;

    const int start_A = block_id * block_sz;
    const int start_B = start_A + sorted_k;
    const int len_A = max(0, min(sorted_k, n - start_A));
    const int len_B = max(0, min(sorted_k, n - start_B));

    if ((int)i - block_id * block_sz >= len_A + len_B) {
        return;
    }

    int id_in_block = i - block_id * block_sz;
    bool from_right = (id_in_block >= len_A);
    uint val = input_data[i];
    if (from_right && start_B >= n) {
        return;
    }

    int count_other;
    if (from_right) {
        count_other = upper_bound(input_data, start_A, len_A, val);
    } else {
        count_other = lower_bound(input_data, start_B, len_B, val);
    }

    int offset_in_side = from_right ? (i - start_B) : (i - start_A);
    int pos = start_A + offset_in_side + count_other;

    output_data[pos] = val;
}
