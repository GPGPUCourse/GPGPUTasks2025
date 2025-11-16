#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int *prefix_sum_accum_gpu,
    const unsigned int *map_result,
    const unsigned int *input_gpu_copy,
    unsigned int *buffer_output_gpu,
    unsigned int n) {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= n) {
        return;
    }

    const unsigned int local_prefix_for_idx = prefix_sum_accum_gpu[idx];
    const unsigned int total_ones = prefix_sum_accum_gpu[n - 1] + map_result[n - 1];

    if (map_result[idx]) {
        const unsigned int position = (n - total_ones) + local_prefix_for_idx;
        buffer_output_gpu[position] = input_gpu_copy[idx];
    } else {
        const unsigned int zero_local_index = idx - local_prefix_for_idx;
        buffer_output_gpu[zero_local_index] = input_gpu_copy[idx];
    }
}

namespace cuda {
    void radix_sort_04_scatter(const gpu::WorkSize &workSize,
                               const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2,
                               gpu::gpu_mem_32u &buffer3, gpu::gpu_mem_32u &st, unsigned int a2) {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), st.cuptr(), a2);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
