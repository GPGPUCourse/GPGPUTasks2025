#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* values, // contains n values
    const unsigned int* per_chunk_counts_prefixes, // will contain chunks_count * CASES_PER_PASS prefix sums (inclusive)
          unsigned int* output_values, // will contain reordered n values
    unsigned int bits_offset,
    unsigned int n)
{
    const unsigned int local_id = threadIdx.x;
    const unsigned int chunk_id = blockIdx.x; // work group index
    const unsigned int chunk_offset = chunk_id * ELEMENTS_PER_CHUNK;
    const unsigned int chunks_count = (n + ELEMENTS_PER_CHUNK - 1) / ELEMENTS_PER_CHUNK; // div_ceil
    const unsigned int last_chunk_id = chunks_count - 1;

    __shared__ unsigned int group_values[ELEMENTS_PER_CHUNK];
    __shared__ unsigned int output_indices[ELEMENTS_PER_CHUNK];
    for (unsigned int i = 0; i < ELEMENTS_PER_WORK_ITEM; ++i) {
        curassert(local_id < WG_SIZE, 435345231);
        const unsigned int global_index = chunk_offset + i * WG_SIZE + local_id;

        //if (global_index == 35) printf("TEST932 %d %d\n", values[global_index], i * WG_SIZE + local_id);
        group_values[i * WG_SIZE + local_id] = (global_index < n) ? values[global_index] : UINT32_INFINITY;
    }

    curassert(CASES_PER_PASS <= WG_SIZE, 473524312);
    __shared__ unsigned int global_cases_personal_offsets[CASES_PER_PASS]; // how many such cases are presented in previous chunks
    __shared__ unsigned int local_cases_count[CASES_PER_PASS];             // how many such cases in current chunk
    __shared__ unsigned int global_cases_offsets[CASES_PER_PASS];          // how many prev cases in total + how many such cases are presented in previous chunks
    if (local_id < CASES_PER_PASS) { // if one of master-threads (each with its case of sorted bits)
        unsigned int case_id = local_id;
        global_cases_personal_offsets[case_id] = (chunk_id == 0) ? 0 : per_chunk_counts_prefixes[(chunk_id - 1) * CASES_PER_PASS + case_id];
        local_cases_count[case_id] = per_chunk_counts_prefixes[chunk_id * CASES_PER_PASS + case_id] - global_cases_personal_offsets[case_id];
        global_cases_offsets[case_id] = 0;
        for (unsigned int prev_case_id = 0; prev_case_id <= case_id; ++prev_case_id) {
            if (prev_case_id < case_id) {
                unsigned int prev_case_global_count = per_chunk_counts_prefixes[last_chunk_id * CASES_PER_PASS + prev_case_id];
                global_cases_offsets[case_id] += prev_case_global_count;
            }
        }
        global_cases_offsets[case_id] += global_cases_personal_offsets[case_id];
    }

    __syncthreads();

    curassert(CASES_PER_PASS <= WG_SIZE, 137894521);
    if (local_id < CASES_PER_PASS) {
        unsigned int case_id = local_id;
        unsigned int case_id_in_chunk_count = 0;
        for (unsigned int i = 0; i < ELEMENTS_PER_CHUNK; ++i) {
            unsigned int value = group_values[i];
            if (((value >> bits_offset) & BITS_PER_PASS_MASK) == case_id) {
                unsigned int output_index = global_cases_offsets[case_id] + case_id_in_chunk_count;
                //if (i == 10) printf("TEST932 %d + %d\n", global_cases_offsets[case_id], case_id_in_chunk_count);
                //if (i == 3) printf("TEST432 %d %d %d\n", case_id, global_cases_offsets[case_id], case_id_in_chunk_count);
                output_indices[i] = output_index;
                ++case_id_in_chunk_count;
                //curassert(case_id_in_chunk_count != 255, 345214321); // check for overflow
            }
        }
    }

    __syncthreads();

    for (unsigned int i = 0; i < ELEMENTS_PER_WORK_ITEM; ++i) {
        curassert(local_id < WG_SIZE, 435345231);
        unsigned int value = group_values[i * WG_SIZE + local_id];
        unsigned int output_index = output_indices[i * WG_SIZE + local_id];
//        if (i * WG_SIZE + local_id == 3) printf("TEST777 output_index=%d\n", output_index);
        output_values[output_index] = value;

        if (output_index < n) {
            //printf("TEST239 output_index=%d value=%d\n", output_index, value);
            output_values[output_index] = value;
        } else {
            //curassert(value == UINT32_INFINITY, 674233124);
        }
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &values, const gpu::gpu_mem_32u &per_chunk_counts_prefixes, gpu::gpu_mem_32u &output_values, unsigned int bits_offset, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(values.cuptr(), per_chunk_counts_prefixes.cuptr(), output_values.cuptr(), bits_offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
