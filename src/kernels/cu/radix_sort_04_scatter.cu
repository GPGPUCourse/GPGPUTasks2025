#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define uint unsigned int

__global__ void radix_sort_04_scatter(
    const uint* input,
    const uint* scan,
          uint* output,
    uint n,
    uint bit)
{
    const uint in_idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (in_idx >= n) return;

    const uint in = input[in_idx];
    const uint digit = ((in >> bit) & 1);

    const uint acc = digit == 0 ? 0 : n - scan[n - 1];
    const uint ones_before_included = scan[in_idx];
    const uint rel_idx = (digit == 0 ? in_idx - ones_before_included : ones_before_included - 1);
    const uint out_idx = acc + rel_idx;

    output[out_idx] = input[in_idx];
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &input, const gpu::gpu_mem_32u &scan, gpu::gpu_mem_32u &output, uint n, uint bit)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), scan.cuptr(), output.cuptr(), n, bit);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
