#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_02_prefix_accumulation(
    __global const unsigned int* partial_sums_in,
    __global const unsigned int* offsets_in,
    __global unsigned int* final_sums_out,
    unsigned int n)
{
    int thread_idx = get_global_id(0);
    int base_global_idx = thread_idx * ELEM_PER_THREAD;

#pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
        int global_idx = base_global_idx + i;
        if (global_idx >= n) break;

        int current_chunk_id = global_idx / CHUNK_SIZE;

        unsigned int offset = 0;
        if (current_chunk_id > 0) {
            offset = offsets_in[current_chunk_id - 1];
        }

        final_sums_out[global_idx] = partial_sums_in[global_idx] + offset;
    }
}