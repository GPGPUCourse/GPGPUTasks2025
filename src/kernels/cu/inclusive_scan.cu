#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include <stdio.h>

__global__ void prefix_sum_01_sum_reduction(
    const unsigned int* pow2_sum, // contains n values
          unsigned int* next_pow2_sum, // will contain (n+1)/2 values
          unsigned int n)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((gx << 1) + 1 >= n) {
        return;
    }
    next_pow2_sum[gx] = pow2_sum[gx << 1] + pow2_sum[(gx << 1) + 1];
}

__global__ void prefix_sum_02_prefix_accumulation(
    const unsigned int* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
          unsigned int* prefix_sum_accum, // we want to make it finally so that prefix_sum_accum[i] = sum[0, i]
          unsigned int n,
          unsigned int pow2)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gx >= n || !(gx & (1u << pow2))) {
        return;
    }
    prefix_sum_accum[gx] +=pow2_sum[(gx >> pow2) - 1];
}

namespace cuda {

void prefix_sum_01_sum_reduction(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &next_pow2_sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_sum_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), next_pow2_sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}

void prefix_sum_02_prefix_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &pow2_sum, gpu::gpu_mem_32u &prefix_sum_accum, unsigned int n, unsigned int pow2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_02_prefix_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(pow2_sum.cuptr(), prefix_sum_accum.cuptr(), n, pow2);
    CUDA_CHECK_KERNEL(stream);
}

void InclusiveScan(
    const gpu::WorkSize& ws,
    gpu::gpu_mem_32u& input_gpu,
    gpu::gpu_mem_32u& buffer1_pow2_sum_gpu,
    gpu::gpu_mem_32u& buffer2_pow2_sum_gpu,
    gpu::gpu_mem_32u& prefix_sum_accum_gpu,
    unsigned int n
) {
    input_gpu.copyToN(buffer1_pow2_sum_gpu, n);
    input_gpu.copyToN(prefix_sum_accum_gpu, n);

    cuda::prefix_sum_02_prefix_accumulation(ws, buffer1_pow2_sum_gpu, prefix_sum_accum_gpu, n, 0);

    for (unsigned int pow2 = 0u; n >> pow2 != 1; pow2++) {
        prefix_sum_01_sum_reduction(gpu::WorkSize(GROUP_SIZE, (n >> (pow2 + 1))), buffer1_pow2_sum_gpu, buffer2_pow2_sum_gpu, n >> pow2);
        std::swap(buffer1_pow2_sum_gpu, buffer2_pow2_sum_gpu);
        prefix_sum_02_prefix_accumulation(ws, buffer1_pow2_sum_gpu, prefix_sum_accum_gpu, n, pow2 + 1);
    }
}

} // namespace cuda
