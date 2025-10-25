#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* reduction_buffer,
    const unsigned int* global_offsets,
    unsigned int* buffer2,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int current_index = index + 1;


    if (index < n) {
        buffer2[index] = 0;
        unsigned int offset = 0;
        for (int i = 27; i >= 0; --i) {
            unsigned int step = 1u << i;
            if (current_index >= step) {
                current_index -= step;
                buffer2[index] += reduction_buffer[global_offsets[i] + offset / step];
                offset += step;
            }
        }
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& buffer1,
    const gpu::gpu_mem_32u& global_offsets,
    gpu::gpu_mem_32u& buffer2,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), global_offsets.cuptr(), buffer2.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
