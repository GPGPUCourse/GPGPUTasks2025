#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"



__global__ void radix_sort_04_scatter(
    const unsigned int* buffer1,   
    const unsigned int* buffer2,   
          unsigned int* buffer3,   
    unsigned int a1,               
    unsigned int shift)               
{


    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned lane = (tid & 31u);
    const unsigned wid = (tid >> 5);             
    const unsigned mask_bins = RADIX_BINS - 1u;
    const unsigned chunk = (a1 + gridDim.x - 1u) / gridDim.x;
    const unsigned L = bid * chunk;
    const unsigned R = min(a1, L + chunk);

    __shared__ unsigned int bin_counter[WARP_BINS_CNT];        
    __shared__ unsigned int warp_prefix[WARP_BINS_CNT];        
    __shared__ unsigned int bin_counter_all_warps[RADIX_BINS]; 
    __shared__ unsigned int block_bin_base[RADIX_BINS];        
    __shared__ unsigned int shared_block_offsets[RADIX_BINS];  

    // инициализация shared
    for (unsigned i = tid; i < WARP_BINS_CNT; i += blockDim.x) {
        bin_counter[i] = 0u;
        warp_prefix[i] = 0u;
    }
    for (unsigned i = tid; i < RADIX_BINS; i += blockDim.x) {
        bin_counter_all_warps[i] = 0u;
        block_bin_base[i]        = 0u;
        shared_block_offsets[i]  = buffer2[bid * RADIX_BINS + i];
    }
    __syncthreads();

    for (unsigned start = L; start < R; start += blockDim.x) {
        for (unsigned i = tid; i < WARP_BINS_CNT; i += blockDim.x) bin_counter[i] = 0u;
        for (unsigned i = tid; i < RADIX_BINS; i += blockDim.x) bin_counter_all_warps[i] = 0u;
        __syncthreads();

        const unsigned idx = start + tid;
        unsigned key = 0u, bin = 0u;
        const bool active = (idx < R);
        if (active) {
            key = buffer1[idx];
            bin = (key >> shift) & mask_bins;
        }

        for (unsigned b = 0; b < RADIX_BINS; ++b) {
            unsigned m = __ballot_sync(0xFFFFFFFFu, active && (bin == b));
            if (active && (bin == b)) {
                if (__ffs(m) == int(lane + 1u)) {
                    bin_counter[wid * RADIX_BINS + b] = __popc(m);
                }
            }
        }
        __syncthreads();

        if (tid < RADIX_BINS) {
            unsigned sum = 0u;
            for (unsigned w = 0; w < WARPS_PER_BLOCK; ++w) {
                const unsigned i = w * RADIX_BINS + tid;
                warp_prefix[i] = sum;     
                sum += bin_counter[i];
            }
            bin_counter_all_warps[tid] = sum;
        }
        __syncthreads();

        if (active) {
            unsigned same_bin_mask = __match_any_sync(0xFFFFFFFFu, bin);
            unsigned intra = __popc(same_bin_mask & ((1u << lane) - 1u));
            unsigned warp_base   = warp_prefix[wid * RADIX_BINS + bin];
            unsigned block_base  = block_bin_base[bin];
            unsigned global_base = shared_block_offsets[bin];

            const unsigned out_idx = global_base + block_base + warp_base + intra;

            if (out_idx < a1) buffer3[out_idx] = key;
        }
        __syncthreads();

        if (tid < RADIX_BINS) {
            block_bin_base[tid] += bin_counter_all_warps[tid];
        }
        __syncthreads();
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &buffer1,     
    const gpu::gpu_mem_32u &buffer2,     
    gpu::gpu_mem_32u &buffer3,           
    unsigned int a1,                     
    unsigned int shift)                      
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, shift);

    CUDA_CHECK_KERNEL(stream);
}
} 
