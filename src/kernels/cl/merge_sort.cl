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
    const unsigned int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    const uint size = 1 << sorted_k;
    const uint block_i = i >> sorted_k;
    const uint pair_start = i & ~((1 << (sorted_k + 1)) - 1);
    if (pair_start + size >= n) {
        output_data[i] = input_data[i];
        return;
    }

    const uint bs_offset = pair_start + (1 - (block_i & 1)) * size;
    uint opp_block_length;
    if (block_i & 1) {
        opp_block_length = size;
    } else {
        opp_block_length = min(size, n - (pair_start + size));
    }

    const uint value = input_data[i] + (block_i & 1);
    uint low = 0;
    uint high = opp_block_length;
    while (low < high) {
        uint mid = (low + high) / 2;
        if (input_data[bs_offset + mid] < value) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    const uint count = low;

    const uint index_in_block = i - (block_i << sorted_k);
    output_data[pair_start + index_in_block + count] = input_data[i];
}
