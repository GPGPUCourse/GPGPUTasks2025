#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* bin_counter,
    unsigned int* bin_base)
{
    static constexpr unsigned int BITS_AT_A_TIME = 4;
    static constexpr unsigned int BINS_CNT = 1u << BITS_AT_A_TIME;

    __shared__ unsigned int bins[BINS_CNT];
    const unsigned int thread_ind = threadIdx.x;
    if (thread_ind < BINS_CNT)
        bins[thread_ind] = bin_counter[thread_ind];
    __syncthreads();

    if (thread_ind == 0) {
        unsigned int acc = 0;
#pragma unroll
        for (unsigned int i = 0; i < BINS_CNT; ++i) {
            unsigned int cur = bins[i];
            bins[i] = acc;
            acc += cur;
        }
    }
    __syncthreads();

    if (thread_ind < BINS_CNT)
        bin_base[thread_ind] = bins[thread_ind];
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(const gpu::gpu_mem_32u& bin_counter, gpu::gpu_mem_32u& bin_base)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<1, WARP_SIZE, 0, stream>>>(bin_counter.cuptr(), bin_base.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
