#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_04_scatter(
    const unsigned int* values,
    const unsigned int* prefixes_scan_accum,
          unsigned int* scatter,
    unsigned int n,
    unsigned int offset)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int digits[GROUP_SIZE];
    if (idx < n) {
        digits[threadIdx.x] = (values[idx] >> (RADIX_BITS * offset)) & RADIX_MASK;
    }
    __syncthreads();

    unsigned int local_offset = 0;

    for (unsigned int i = 0; i < threadIdx.x; i++) {
        if (digits[threadIdx.x] == digits[i]) {
            local_offset++;
        }
    }
    
    if (idx < n) {
        unsigned int pref_pos = digits[threadIdx.x] * (n + GROUP_SIZE - 1) / GROUP_SIZE + blockIdx.x;
        unsigned int global_offset = pref_pos > 0 ? prefixes_scan_accum[pref_pos - 1] : 0;
        scatter[global_offset + local_offset] = values[idx];
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1, const gpu::gpu_mem_32u &buffer2, gpu::gpu_mem_32u &buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
