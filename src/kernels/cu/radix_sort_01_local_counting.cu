#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* in,
    unsigned int* block_hist,
    unsigned int n,
    unsigned int shift)
{
    const unsigned int thread_ind = threadIdx.x, block_ind = blockIdx.x;
    const unsigned int chunk_size = (n + gridDim.x - 1u) / gridDim.x;
    const unsigned int block_l = block_ind * chunk_size;
    const unsigned int block_r = min(n, block_l + chunk_size);

    __shared__ unsigned int warp_hist[WARP_BINS_CNT];
    if (thread_ind < WARP_BINS_CNT)
        warp_hist[thread_ind] = 0;
    __syncthreads();

    const unsigned int warp_base = (thread_ind >> 5) << BITS_AT_A_TIME;
    for (unsigned int i = block_l + thread_ind; i < block_r; i += BLOCK_THREADS)
        atomicAdd(&warp_hist[warp_base + ((in[i] >> shift) & (BINS_CNT - 1u))], 1u);
    __syncthreads();

    if (thread_ind < BINS_CNT) {
        unsigned int bin_sum = 0;
        for (unsigned int i = 0; i < WARPS_CNT; ++i)
            bin_sum += warp_hist[thread_ind + (i << BITS_AT_A_TIME)];
        block_hist[thread_ind + (block_ind << BITS_AT_A_TIME)] = bin_sum;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::gpu_mem_32u& in, gpu::gpu_mem_32u& block_hist, const unsigned int& n, const unsigned int& shift, const unsigned int& blocks_cnt)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<blocks_cnt, BLOCK_THREADS, 0, stream>>>(in.cuptr(), block_hist.cuptr(), n, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
