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
    __global const uint4* partial_sums_in_v4 = (__global const uint4*)partial_sums_in;
    __global uint4* final_sums_out_v4 = (__global uint4*)final_sums_out;

    int thread_idx = get_global_id(0);
    int base_global_idx = thread_idx * ELEM_PER_THREAD;

    int vectorized_limit = (ELEM_PER_THREAD / VEC_SIZE) * VEC_SIZE;

#pragma unroll
    for (int i = 0; i < vectorized_limit; i += VEC_SIZE) {
        int current_scalar_idx = base_global_idx + i;

        if (current_scalar_idx + VEC_SIZE - 1 < n) {
            int vec_idx = current_scalar_idx / VEC_SIZE;

            uint4 partial_sum = partial_sums_in_v4[vec_idx];

            int current_chunk_id = current_scalar_idx / CHUNK_SIZE;
            unsigned int offset_val = 0;
            if (current_chunk_id > 0) {
                offset_val = offsets_in[current_chunk_id - 1];
            }
            uint4 offset_vec = (uint4)(offset_val);

            final_sums_out_v4[vec_idx] = partial_sum + offset_vec;
        } else {
            for (int j = 0; j < VEC_SIZE; ++j) {
                 int global_idx = current_scalar_idx + j;
                 if (global_idx >= n) break;

                 int chunk_id = global_idx / CHUNK_SIZE;
                 unsigned int offset = (chunk_id > 0) ? offsets_in[chunk_id - 1] : 0;
                 final_sums_out[global_idx] = partial_sums_in[global_idx] + offset;
            }
        }
    }

    for (int i = vectorized_limit; i < ELEM_PER_THREAD; i++) {
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