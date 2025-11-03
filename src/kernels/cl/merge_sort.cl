#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   uint  sorted_k,
                   uint  n)
{
    const unsigned int idx = get_global_id(0);
    if (idx >= n) {
        return;
    }

    const unsigned int my_elem = input_data[idx];

    const unsigned int num_buckets = (n + sorted_k - 1) / sorted_k;
    const unsigned int my_bucket_id = idx / sorted_k;
    const unsigned int neighbour_bucket_id = (my_bucket_id % 2 == 0) ? (my_bucket_id + 1) : (my_bucket_id - 1);

    if (neighbour_bucket_id >= num_buckets) {
        output_data[idx] = my_elem;
        return;
    }

    const uint nb_base = neighbour_bucket_id * sorted_k;
    int left = 0;
    int right = min(sorted_k, n - nb_base) - 1;
    while (left <= right) {
        int cur_idx = (right + left) / 2;
        // так как мы работаем с целыми числами, то в целом можем везде оставить <=, если сделать где нужно -1,
        // а так как у нас мин. значение = 1, то это не приведет к переполнению
        if (input_data[nb_base + cur_idx] <= my_elem - (my_bucket_id % 2)) {
            left = cur_idx + 1;
        } else {
            right = cur_idx - 1;
        }
    }

    if (my_bucket_id % 2 == 0) {
        output_data[my_bucket_id * sorted_k + idx % sorted_k + left] = my_elem;
    } else {
        output_data[neighbour_bucket_id * sorted_k + idx % sorted_k + left] = my_elem;
    }

}
