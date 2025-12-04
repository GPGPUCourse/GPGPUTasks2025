# 0 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/merge_sort.cl"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/merge_sort.cl"




# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/rassert.cl" 1
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/../../defines.h" 1
# 2 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/helpers/rassert.cl" 2
# 6 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/merge_sort.cl" 2
# 1 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/../defines.h" 1
# 7 "/home/mikhail/GPGPUTasks2025/GPGPUTasks2025/src/kernels/cl/merge_sort.cl" 2

static inline int lower_bound(__global const uint* arr, int base, int len, uint value) {
    int l = 0, h = len;
    while (l < h) {
        int mid = (l + h) / 2;
        if (arr[base + mid] < value)
            l = mid + 1;
        else
            h = mid;
    }
    return l;
}

static inline int upper_bound(__global const uint* arr, int base, int len, uint value) {
    int l = 0, h = len;
    while (l < h) {
        int mid = (l + h) / 2;
        if (arr[base + mid] <= value)
            l = mid + 1;
        else
            h = mid;
    }
    return l;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
                   int sorted_k,
                   int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) return;

    const int block_size = 2 * sorted_k;
    const int block_id = i / block_size;

    const int left_start = block_id * block_size;
    const int right_start = left_start + sorted_k;

    const int left_len = (left_start < n) ? min(sorted_k, n - left_start) : 0;
    const int right_len = (right_start < n) ? min(sorted_k, n - right_start) : 0;

    const int pos_in_block = i - left_start;


    const uint value = input_data[i];

    if (pos_in_block < left_len) {
        int count = lower_bound(input_data, right_start, right_len, value);
        output_data[left_start + pos_in_block + count] = value;
    } else {
        int count = upper_bound(input_data, left_start, left_len, value);
        int pos_in_right = pos_in_block - left_len;
        output_data[left_start + pos_in_right + count] = value;
    }
}
