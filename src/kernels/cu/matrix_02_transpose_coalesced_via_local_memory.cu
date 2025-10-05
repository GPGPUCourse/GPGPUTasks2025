#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define LOC_SIZE_X (GROUP_SIZE_X + 1) // didn't speed up much, though
#define LOC_SIZE_Y GROUP_SIZE_X

// precondition: blockDim.x == blockDim.y
__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    unsigned int sz = GROUP_SIZE_X; // = GROUP_SIZE_Y

    __shared__ float local_data[LOC_SIZE_X * LOC_SIZE_Y];

    // copy from input (vram) to local shared memory
    unsigned int from_col = blockIdx.x * sz + threadIdx.x;
    unsigned int from_row = blockIdx.y * sz + threadIdx.y;

    if (from_row < h && from_col < w) {
        local_data[threadIdx.y * LOC_SIZE_X + threadIdx.x] = matrix[from_row * w + from_col];
    }

    __syncthreads();

    // copy from local memory to output (vram)
    unsigned int to_col = blockIdx.y * sz + threadIdx.x;
    unsigned int to_row = blockIdx.x * sz + threadIdx.y;

    if (to_row < w && to_col < h) {
        transposed_matrix[to_row * h + to_col] = local_data[threadIdx.x * LOC_SIZE_X + threadIdx.y];
    }
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
