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
    const uint index = get_global_id(0);
    if (index < n) {
        int block_size = 2 * sorted_k;
        int block_id = index / block_size;
        int start_a = block_id * block_size;
        int start_b = start_a + sorted_k;
        int len_a = min(sorted_k, n - start_a);
        int len_b = min(sorted_k, n - start_b);
        if (len_a <= 0 && len_b <= 0) {
            return;
        }
        uint val = input_data[index];
        bool second = (index - start_a) >= len_a;
        int offset;
        if (second) {
            offset = index - start_b;
        } else {
            offset = index - start_a;
        }
        int l = 0;
        int r = second ? len_a : len_b;
        while (l < r) {
            int m = (l + r) >> 1;
            uint cmp_val = input_data[(second ? start_a : start_b) + m];
            if ((second && cmp_val <= val) || (!second && cmp_val < val)) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        int pos = start_a + offset + l;
        if (pos < n) {
            output_data[pos] = val;
        }
    }
}
