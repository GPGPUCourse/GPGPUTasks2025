#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

#define LOCAL_MEM_BANKS 32
#define PADDED_INDEX(idx) ((idx) + (idx) / LOCAL_MEM_BANKS)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
prefix_sum_01_reduction(
    __global const unsigned int* input,
    __global unsigned int* partial_sums_out,
    __global unsigned int* group_sums_out,
    unsigned int n)
{
    __local unsigned int local_data_buffer[CHUNK_SIZE];
    __local unsigned int local_sums_buffer[PADDED_INDEX(GROUP_SIZE)];

    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    int global_base_idx_striped = group_id * CHUNK_SIZE + local_id;
#pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
        unsigned long long global_idx = global_base_idx_striped + i * GROUP_SIZE;
        local_data_buffer[local_id + i * GROUP_SIZE] = (global_idx < n) ? input[global_idx] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#if ELEM_PER_THREAD == 4
    __local uint4* local_data_as_uint4 = (__local uint4*)local_data_buffer;
    uint4 my_elements_vec = local_data_as_uint4[local_id];

    my_elements_vec.y += my_elements_vec.x;
    my_elements_vec.z += my_elements_vec.y;
    my_elements_vec.w += my_elements_vec.z;

    unsigned int running_sum = my_elements_vec.w;
#else
    int local_base_idx_sequential = local_id * ELEM_PER_THREAD;
    unsigned int my_elements[ELEM_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
        my_elements[i] = local_data_buffer[local_base_idx_sequential + i];
    }

    unsigned int running_sum = 0;
#pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
        running_sum += my_elements[i];
        my_elements[i] = running_sum;
    }
#endif

    local_sums_buffer[PADDED_INDEX(local_id)] = running_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    int offset = 1;
    for (int d = GROUP_SIZE >> 1; d > 0; d >>= 1) {
        if (local_id < d) {
            int ai = PADDED_INDEX(offset * (2 * local_id + 1) - 1);
            int bi = PADDED_INDEX(offset * (2 * local_id + 2) - 1);
            local_sums_buffer[bi] += local_sums_buffer[ai];
        }
        offset *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        group_sums_out[group_id] = local_sums_buffer[PADDED_INDEX(GROUP_SIZE - 1)];
        local_sums_buffer[PADDED_INDEX(GROUP_SIZE - 1)] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d = 1; d < GROUP_SIZE; d *= 2) {
        offset >>= 1;
        if (local_id < d) {
            int ai = PADDED_INDEX(offset * (2 * local_id + 1) - 1);
            int bi = PADDED_INDEX(offset * (2 * local_id + 2) - 1);
            unsigned int temp = local_sums_buffer[ai];
            local_sums_buffer[ai] = local_sums_buffer[bi];
            local_sums_buffer[bi] += temp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    unsigned int thread_offset = local_sums_buffer[PADDED_INDEX(local_id)];

#if ELEM_PER_THREAD == 4
    my_elements_vec += (uint4)(thread_offset);
    local_data_as_uint4[local_id] = my_elements_vec;
#else
#pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
        local_data_buffer[local_base_idx_sequential + i] = my_elements[i] + thread_offset;
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int i = 0; i < ELEM_PER_THREAD; i++) {
        int global_idx = global_base_idx_striped + i * GROUP_SIZE;
        if (global_idx < n) {
            partial_sums_out[global_idx] = local_data_buffer[local_id + i * GROUP_SIZE];
        }
    }
}