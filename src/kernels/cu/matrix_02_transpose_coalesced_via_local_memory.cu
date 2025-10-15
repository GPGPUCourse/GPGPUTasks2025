#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_transpose_coalesced_via_local_memory(
    const float* input,
    float* output,
    unsigned int width,
    unsigned int height)
{
    __shared__ float block_data[GROUP_SIZE + GROUP_SIZE_Y];

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    float elm = 0.0f;
    if (col < width && row < height) {
        elm = input[row * width + col];
    }

    int store_pos = (blockDim.x + 1) * threadIdx.y + threadIdx.x;
    if (store_pos < (GROUP_SIZE + GROUP_SIZE_Y)) {
        block_data[store_pos] = elm;
    }

    __syncthreads();

    unsigned int trans_col = blockIdx.y * blockDim.y + threadIdx.x;
    unsigned int trans_row = blockIdx.x * blockDim.x + threadIdx.y;

    if (trans_col < height && trans_row < width) {
        int load_pos = (blockDim.y + 1) * threadIdx.x + threadIdx.y;
        if (load_pos < (GROUP_SIZE + GROUP_SIZE_Y)) {
            output[trans_row * height + trans_col] = block_data[load_pos];
        }
    }
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& matrix, gpu::gpu_mem_32f& transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
