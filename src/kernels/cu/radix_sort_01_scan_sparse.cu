#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define uint unsigned int

__global__ void radix_sort_01_scan_sparse(
    const uint* input, // size = n
    uint* buffer, // size = n / GROUP_SIZE + 1
    uint* output, // size = n
    uint n,
    uint bit)
{
    __shared__ uint part[GROUP_SIZE];

    const uint in_idx = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint loc_idx = threadIdx.x;
    const uint out_idx = in_idx / GROUP_SIZE;

    if (in_idx < n) {
        part[loc_idx] = (input[in_idx] >> bit) & 1;
    } else {
        part[loc_idx] = 0;
    }

    __syncthreads();

    uint sum = 0;
    if (loc_idx == 0) {
        for (size_t i = 0; i < GROUP_SIZE; ++i) {
            sum += part[i];
            output[blockIdx.x * blockDim.x + i] = sum;
        }
        buffer[out_idx] = sum;
    }
}

namespace cuda {
void radix_sort_01_scan_sparse(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& buffer, gpu::gpu_mem_32u& output, uint n, uint bit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_scan_sparse<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), buffer.cuptr(), output.cuptr(), n, bit);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda