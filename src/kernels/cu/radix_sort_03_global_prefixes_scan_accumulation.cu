#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    const unsigned int* buffer1,   
          unsigned int* buffer2,  
    unsigned int a1,               
    unsigned int a2)              
{
    (void)a2;
    const unsigned tid = threadIdx.x;

    if (blockIdx.x != 0) return;

    __shared__ unsigned int totals[RADIX_BINS];
    __shared__ unsigned int bin_base[RADIX_BINS];

    for (unsigned bin = tid; bin < RADIX_BINS; bin += blockDim.x) {
        unsigned sum = 0u;
        for (unsigned b = 0; b < a1; ++b) {
            sum += buffer1[b * RADIX_BINS + bin];
        }
        totals[bin] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        unsigned acc = 0u;
        for (unsigned bin = 0; bin < RADIX_BINS; ++bin) {
            const unsigned cur = totals[bin];
            bin_base[bin] = acc;
            acc += cur;
        }
    }
    __syncthreads();

    for (unsigned bin = tid; bin < RADIX_BINS; bin += blockDim.x) {
        const unsigned base = bin_base[bin];
        for (unsigned b = 0; b < a1; ++b) {
            const unsigned idx = b * RADIX_BINS + bin;
            buffer2[idx] += base;
        }
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &buffer1, gpu::gpu_mem_32u &buffer2, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda