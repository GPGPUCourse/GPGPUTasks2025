#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>


#include <libgpu/cuda/cu/common.cu>
#include <vector_types.h>

#include "../defines.h"
#include "helpers/rassert.cu"


__global__ void radix_sort_01_local_counting(
    const unsigned int* arr,
    unsigned int* buffer2,
    unsigned int bitCount,
    unsigned int offset,
    unsigned int n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int mask = (1u << bitCount) - 1u;

    // __shared__ unsigned int local_counts[256];


    // if (index < (1u << bitCount)) {
    //     local_counts[index] = 0;
    // }

    // __syncthreads();

    if (index < n) {
        unsigned int value = arr[index];
        unsigned int bit = (value >> offset) & mask;
        atomicAdd(&buffer2[bit], 1u);
    }

    // __syncthreads();

    // if (index < (1u << bitCount)) {
    //     atomicAdd(&buffer2[index], local_counts[index]);
    // }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& buffer1,
    gpu::gpu_mem_32u& buffer2,
    unsigned int bitCount,
    unsigned int offset,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), bitCount, offset, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
