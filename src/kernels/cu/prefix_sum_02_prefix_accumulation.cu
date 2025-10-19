#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int* array,
    const unsigned int* global_offset,
    const unsigned int* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
          unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
    unsigned int n,
    unsigned int pow2)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int current_index = index + 1;

    prefix_sum_accum[index] = 0;

    if (index < n) {
        unsigned int offset = 0;
        for (int i = pow2; i > 0; --i) {
            unsigned int step = 1u << i;
            if (current_index >= step) {
                current_index -= step;
                prefix_sum_accum[index] += pow2_sum[global_offset[i] + offset / step];
                offset += step;
            }
        }

        if (current_index > 0) {
            prefix_sum_accum[index] += array[index];
        }
    }
}

namespace cuda {
void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u& array,
            const gpu::gpu_mem_32u& global_offset,
            const gpu::gpu_mem_32u &pow2_sum,
            gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(array.cuptr(), global_offset.cuptr(), pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
