#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"
#include <cub/cub.cuh>

__global__ void excl_block_scan(const unsigned int* in, unsigned int* out, unsigned int* block_sums, unsigned int n)
{
    const unsigned int thread_ind = threadIdx.x;
    const unsigned int lane_ind = thread_ind & (WARP_SIZE - 1);
    const unsigned int warp_ind = thread_ind >> 5;
    const unsigned int block_ind = blockIdx.x;
    const unsigned int block_start_ind = block_ind * BLOCK_ELEMS;
    const unsigned int lane_mask = __activemask();

    __shared__ unsigned int warp_totals[WARPS_CNT];
    __shared__ unsigned int block_sum;

    unsigned int vals[THREAD_ELEMS];
#pragma unroll
    for (unsigned int i = 0; i < THREAD_ELEMS; ++i) {
        const unsigned int ind = block_start_ind + thread_ind + i * BLOCK_THREADS;
        vals[i] = (ind < n) ? __ldg(in + ind) : 0u;
    }

    unsigned int block_total = 0;
#pragma unroll
    for (unsigned int i = 0; i < THREAD_ELEMS; ++i) {
        const unsigned int ind = block_start_ind + thread_ind + i * BLOCK_THREADS;
        const unsigned int val = vals[i];
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
        if (ind < n)
            out[ind] = block_total + block_excl_sum;

        if (thread_ind == BLOCK_THREADS - 1u)
            block_sum = block_excl_sum + val;
        __syncthreads();

        block_total += block_sum;
    }

    if (block_sums && thread_ind == BLOCK_THREADS - 1)
        block_sums[block_ind] = block_total;
}

__global__ void acc(unsigned int* data, const unsigned int* block_offsets, unsigned int n)
{
    const unsigned int block_ind = blockIdx.x;
    const unsigned int block_start_ind = block_ind * BLOCK_ELEMS;
    if (block_start_ind >= n)
        return;

    const unsigned int thread_ind = threadIdx.x;
    const unsigned int block_offset = __ldg(block_offsets + block_ind);
#pragma unroll
    for (unsigned int i = 0; i < THREAD_ELEMS; ++i) {
        const unsigned int ind = block_start_ind + thread_ind + i * BLOCK_THREADS;
        if (ind < n)
            data[ind] += block_offset;
    }
}

namespace cuda {
void prefix_sum(const unsigned int* in, unsigned int* out, const unsigned int& n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    const unsigned int blocks_cnt = (n + BLOCK_ELEMS - 1u) / BLOCK_ELEMS;
    gpu::gpu_mem_32u block_sums(blocks_cnt);
    ::excl_block_scan<<<blocks_cnt, BLOCK_THREADS, 0, stream>>>(in, out, block_sums.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);

    if (blocks_cnt > 1) {
        gpu::gpu_mem_32u block_offsets(blocks_cnt);
        prefix_sum(block_sums.cuptr(), block_offsets.cuptr(), blocks_cnt);
        ::acc<<<blocks_cnt, BLOCK_THREADS, 0, stream>>>(out, block_offsets.cuptr(), n);
        CUDA_CHECK_KERNEL(stream);
    }
}
} // namespace cuda