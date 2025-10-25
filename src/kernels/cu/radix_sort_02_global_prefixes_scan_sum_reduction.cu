#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

static constexpr unsigned int BLOCK_THREADS = 256;
static constexpr unsigned int THREAD_ELEMS = 32;
static constexpr unsigned int BLOCK_ELEMS = BLOCK_THREADS * THREAD_ELEMS;
static constexpr unsigned int WARPS_CNT = BLOCK_THREADS >> 5;
static constexpr unsigned int BITS_AT_A_TIME = 4;
static constexpr unsigned int BINS_CNT = 1u << BITS_AT_A_TIME;

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(const unsigned int* in, unsigned int* out, unsigned int* block_sums, unsigned int n)
{
    const unsigned int thread_ind = threadIdx.x;
    const unsigned int lane_ind = thread_ind & (WARP_SIZE - 1);
    const unsigned int warp_ind = thread_ind >> 5;
    const unsigned int block_ind = blockIdx.x;
    const unsigned int lane_mask = __activemask();

    __shared__ unsigned int warp_totals[WARPS_CNT];
    __shared__ unsigned int block_sum;

    unsigned int total = 0;
    for (unsigned int base = 0; base < n; base += BLOCK_ELEMS) {
        unsigned int block_total = 0;
#pragma unroll
        for (unsigned int i = 0; i < THREAD_ELEMS; ++i) {
            const unsigned int bin_ind = base + thread_ind + i * BLOCK_THREADS;
            const unsigned int ind = (bin_ind << BITS_AT_A_TIME) + block_ind;
            const unsigned int val = (bin_ind < n) ? __ldg(in + ind) : 0u;
            unsigned int incl_sum = val;

            unsigned int tmp;
#pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
                tmp = __shfl_up_sync(lane_mask, incl_sum, offset);
                if (lane_ind >= offset)
                    incl_sum += tmp;
            }

            if (lane_ind == WARP_SIZE - 1)
                warp_totals[warp_ind] = incl_sum;
            __syncthreads();

            if (warp_ind == 0) {
                const unsigned int warp_val = (lane_ind < WARPS_CNT) ? warp_totals[lane_ind] : 0u;
                unsigned int warp_incl_sum = warp_val;

#pragma unroll
                for (int offset = 1; offset < WARPS_CNT; offset <<= 1) {
                    tmp = __shfl_up_sync(lane_mask, warp_incl_sum, offset, WARPS_CNT);
                    if (lane_ind >= offset)
                        warp_incl_sum += tmp;
                }

                if (lane_ind < WARPS_CNT)
                    warp_totals[lane_ind] = warp_incl_sum - warp_val;
            }
            __syncthreads();

            const unsigned int warp_offset = warp_totals[warp_ind];
            const unsigned int block_excl_sum = warp_offset + incl_sum - val;
            if (bin_ind < n)
                out[ind] = block_total + block_excl_sum + total;

            if (thread_ind == BLOCK_THREADS - 1u)
                block_sum = block_excl_sum + val;
            __syncthreads();

            block_total += block_sum;
        }

        total += block_total;
    }

    if (thread_ind == BLOCK_THREADS - 1u)
        block_sums[block_ind] = total;
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::gpu_mem_32u& block_hist, gpu::gpu_mem_32u& block_offsets, gpu::gpu_mem_32u& bin_counter, const unsigned int& blocks_cnt)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<BINS_CNT, BLOCK_THREADS, 0, stream>>>(block_hist.cuptr(), block_offsets.cuptr(), bin_counter.cuptr(), blocks_cnt);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
