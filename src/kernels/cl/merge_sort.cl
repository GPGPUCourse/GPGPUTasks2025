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

    const uint chunk_id = i / sorted_k;
    const uint chunk_offset = chunk_id * sorted_k;

    // printf("[%d] chunk_id=%d chunk_offset=%d\n", i, chunk_id, chunk_offset);

    const uint pair_chunk_id = chunk_id + (chunk_id % 2 == 1 ? -1 : 1);
    const uint pair_chunk_offset = pair_chunk_id * sorted_k;

    // printf("[%d] pair_chunk_id=%d pair_chunk_offset=%d\n", i, pair_chunk_id, pair_chunk_offset);

    uint l = pair_chunk_offset, r = min(pair_chunk_offset + sorted_k, n);
    uint target = input_data[i];
    while (l < r) {
        uint mid = (l + r) / 2;
        if (input_data[mid] < target && chunk_id % 2 == 0 || input_data[mid] <= target && chunk_id % 2 == 1) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }

    // printf("[%d] l=%d r=%d target=%d\n", i, l, r, target);

    uint new_index = l - pair_chunk_offset + i - chunk_offset;
    // printf("[%d] new_index=%d\n", i, new_index);
    output_data[min(chunk_offset, pair_chunk_offset) + new_index] = target;
}
