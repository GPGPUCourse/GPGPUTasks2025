#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_04_scatter(
    const unsigned int* input_gpu,
    const unsigned int* scan_buffer_gpu,
    unsigned int* output_gpu,
    unsigned int bitCount,
    unsigned int offset,
    unsigned int local_offset,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int fullMask = (1u << bitCount) - 1u;

    if (index < n) {
        if (index == 0 && scan_buffer_gpu[0] == 1 || index > 0 && scan_buffer_gpu[index] > scan_buffer_gpu[index - 1]) {
            unsigned int value = input_gpu[index];
            unsigned int mask = (value >> offset) & fullMask;
            unsigned int position = local_offset + scan_buffer_gpu[index] - 1;

            output_gpu[position] = value;
        }
    }
}

namespace cuda {
void radix_sort_04_scatter(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_gpu,
    const gpu::gpu_mem_32u& scan_buffer_gpu,
    gpu::gpu_mem_32u& output_gpu,
    unsigned int bitCount,
    unsigned int offset,
    unsigned int local_offset,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_gpu.cuptr(), scan_buffer_gpu.cuptr(), output_gpu.cuptr(), bitCount, offset, local_offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
