#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_transpose_naive(
    const float* matrix, // w x h
    float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < w && j < h) {
        auto el = matrix[j * w + i];
        transposed_matrix[i * h + j] = el;
    }
}

namespace cuda {
void matrix_transpose_naive(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& matrix, gpu::gpu_mem_32f& transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_naive<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
