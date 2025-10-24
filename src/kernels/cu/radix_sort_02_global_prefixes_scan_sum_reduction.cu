#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "../kernels.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(const unsigned int* block_hist, const unsigned int* block_offsets, unsigned int* bin_counter, unsigned int blocks_cnt)
{
    const unsigned int thread_ind = threadIdx.x;
    if (thread_ind >= BINS_CNT)
        return;
    bin_counter[thread_ind] = __ldg(block_hist + thread_ind * blocks_cnt + blocks_cnt - 1u) + __ldg(block_offsets + thread_ind * blocks_cnt + blocks_cnt - 1u);
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::gpu_mem_32u& block_hist, gpu::gpu_mem_32u& block_offsets, gpu::gpu_mem_32u& bin_counter, const unsigned int& blocks_cnt)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    const unsigned int* in = block_hist.cuptr();
    unsigned int* out = block_offsets.cuptr();
#pragma unroll
    for (unsigned int i = 0; i < BINS_CNT; ++i) {
        prefix_sum(in, out, blocks_cnt);
        in += blocks_cnt;
        out += blocks_cnt;
    }

    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<1, WARP_SIZE, 0, stream>>>(block_hist.cuptr(), block_offsets.cuptr(), bin_counter.cuptr(), blocks_cnt);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
