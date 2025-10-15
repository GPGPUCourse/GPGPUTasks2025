#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_multiply_via_local_memory(
    const float* input_matrix_A,
    const float* input_matrix_B,
    float* result_matrix_C,
    unsigned int matrix_width,
    unsigned int matrix_height,
    unsigned int inner_dimension)
{

    extern __shared__ float shared_memory_buffer[];

    float* tile_buffer_A = shared_memory_buffer;
    float* tile_buffer_B = shared_memory_buffer + GROUP_SIZE;

    const unsigned int global_row = blockIdx.y * GROUP_SIZE_Y + threadIdx.y;
    const unsigned int global_col = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    const unsigned int tile_index = threadIdx.y * GROUP_SIZE_X + threadIdx.x;

    float sum = 0.0f;

    for (unsigned int tile_offset = 0; tile_offset < inner_dimension; tile_offset += GROUP_SIZE_X) {

        unsigned int col_A = tile_offset + threadIdx.x;
        if (global_row < matrix_height && col_A < inner_dimension) {
            tile_buffer_A[tile_index] = input_matrix_A[global_row * inner_dimension + col_A];
        } else {
            tile_buffer_A[tile_index] = 0.0f;
        }

        unsigned int row_B = tile_offset + threadIdx.y;
        if (row_B < inner_dimension && global_col < matrix_width) {
            tile_buffer_B[tile_index] = input_matrix_B[row_B * matrix_width + global_col];
        } else {
            tile_buffer_B[tile_index] = 0.0f;
        }

        __syncthreads();

        for (unsigned int k = 0; k < GROUP_SIZE_X; k++) {
            unsigned int idx_A = threadIdx.y * GROUP_SIZE_X + k;
            unsigned int idx_B = k * GROUP_SIZE_X + threadIdx.x;
            sum += tile_buffer_A[idx_A] * tile_buffer_B[idx_B];
        }

        __syncthreads();
    }

    if (global_col < matrix_width && global_row < matrix_height) {
        result_matrix_C[global_row * matrix_width + global_col] = sum;
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    unsigned int block_size = workSize.cuBlockSize().x * workSize.cuBlockSize().y;
    unsigned int shared_mem_size = 2 * block_size * sizeof(float);

    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), shared_mem_size, stream>>>(
        a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);

    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
