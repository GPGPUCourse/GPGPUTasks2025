#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

static constexpr unsigned int BLOCK_THREADS = 256;
static constexpr unsigned int WARPS_CNT = BLOCK_THREADS >> 5;
static constexpr unsigned int BITS_AT_A_TIME = 4;
static constexpr unsigned int BINS_CNT = 1u << BITS_AT_A_TIME;
static constexpr unsigned int WARP_BINS_CNT = WARPS_CNT << BITS_AT_A_TIME;

__global__ void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    const unsigned int* in,
    const unsigned int* block_offsets,
    const unsigned int* bin_base,
    unsigned int* out,
    unsigned int n,
    unsigned int shift)
{
    const unsigned int thread_ind = threadIdx.x, block_ind = blockIdx.x;
    const unsigned int warp_base = (thread_ind >> 5) << BITS_AT_A_TIME;
    const unsigned int lane_ind = thread_ind & (WARP_SIZE - 1);
    const unsigned int chunk_size = (n + gridDim.x - 1u) / gridDim.x;
    const unsigned int block_l = block_ind * chunk_size;
    const unsigned int block_r = min(n, block_l + chunk_size);

    __shared__ unsigned int bin_counter[WARP_BINS_CNT];
    __shared__ unsigned int warp_pref[WARP_BINS_CNT];
    __shared__ unsigned int bin_counter_by_all_warps[BINS_CNT];
    __shared__ unsigned int block_bin_base[BINS_CNT];
    __shared__ unsigned int shared_block_offsets[BINS_CNT];
    __shared__ unsigned int shared_bin_base[BINS_CNT];
    if (thread_ind < BINS_CNT) {
        block_bin_base[thread_ind] = 0;
        shared_block_offsets[thread_ind] = __ldg(block_offsets + thread_ind + (block_ind << BITS_AT_A_TIME));
        shared_bin_base[thread_ind] = __ldg(bin_base + thread_ind);
    }
    __syncthreads();

    for (unsigned int start = block_l; start < block_r; start += BLOCK_THREADS) {
        if (thread_ind < WARP_BINS_CNT)
            bin_counter[thread_ind] = 0;
        if (thread_ind < BINS_CNT)
            bin_counter_by_all_warps[thread_ind] = 0;
        __syncthreads();

        const unsigned int ind = start + thread_ind;
        unsigned int val = 0, bin = 0;
        if (ind < block_r) {
            val = in[ind];
            bin = (val >> shift) & (BINS_CNT - 1u);
        }

        unsigned int masks[BINS_CNT];
#pragma unroll
        for (unsigned int b = 0; b < BINS_CNT; ++b)
            masks[b] = __ballot_sync(__activemask(), (ind < block_r) && (bin == b));
        const unsigned int mask = masks[bin];
        if ((ind < block_r) && (__ffs(mask) == (lane_ind + 1u)))
            bin_counter[bin + warp_base] = __popc(mask);
        __syncthreads();

        if (thread_ind < BINS_CNT) {
            unsigned int sum = 0;
#pragma unroll
            for (unsigned int w = 0; w < WARPS_CNT; ++w) {
                const unsigned int i = (w << BITS_AT_A_TIME) + thread_ind;
                warp_pref[i] = sum;
                sum += bin_counter[i];
            }
            bin_counter_by_all_warps[thread_ind] = sum;
        }
        __syncthreads();

        if (ind < block_r)
            out[shared_bin_base[bin] + shared_block_offsets[bin] + block_bin_base[bin] + warp_pref[bin + warp_base] + __popc(mask & ((1u << lane_ind) - 1u))] = val;
        __syncthreads();

        if (thread_ind < BINS_CNT)
            block_bin_base[thread_ind] += bin_counter_by_all_warps[thread_ind];
        __syncthreads();
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::gpu_mem_32u& in, const gpu::gpu_mem_32u& block_offsets, const gpu::gpu_mem_32u& bin_base, gpu::gpu_mem_32u& out, const unsigned int& n, const unsigned int& shift, const unsigned int& blocks_cnt)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<blocks_cnt, BLOCK_THREADS, 0, stream>>>(in.cuptr(), block_offsets.cuptr(), bin_base.cuptr(), out.cuptr(), n, shift);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
