#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* values, // contains n values
    // TODO try char
          unsigned int* per_chunk_counts, // will contain chunks_count * CASES_PER_PASS
    unsigned int bits_offset,
    unsigned int n)
{
    const unsigned int local_id = threadIdx.x;
    const unsigned int group_id = blockIdx.x;
    const unsigned int chunk_id = group_id;
    const unsigned int chunk_offset = chunk_id * ELEMENTS_PER_CHUNK;

    __shared__ unsigned int group_values[ELEMENTS_PER_CHUNK];
    for (unsigned int i = 0; i < ELEMENTS_PER_WORK_ITEM; ++i) {
        curassert(local_id < WG_SIZE, 435345231);
        const unsigned int global_index = chunk_offset + i * WG_SIZE + local_id;
        group_values[i * WG_SIZE + local_id] = (global_index < n) ? values[global_index] : UINT32_INFINITY;
    }

    __syncthreads();

    curassert(CASES_PER_PASS <= WG_SIZE, 43524312);
    if (local_id < CASES_PER_PASS) { // if one of master-threads (each with its case of sorted bits)
        unsigned int case_id = local_id;
        unsigned int case_id_in_chunk_count = 0;
        for (unsigned int i = 0; i < ELEMENTS_PER_CHUNK; ++i) {
            unsigned int value = group_values[i];
            if (((value >> bits_offset) & BITS_PER_PASS_MASK) == case_id) {
                ++case_id_in_chunk_count;
                //curassert(case_id_in_chunk_count != 255, 345214321); // check for overflow
            }
        }
        per_chunk_counts[chunk_id * CASES_PER_PASS + case_id] = case_id_in_chunk_count;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &values, gpu::gpu_mem_32u &per_chunk_counts, unsigned int bits_offset, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(values.cuptr(), per_chunk_counts.cuptr(), bits_offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
