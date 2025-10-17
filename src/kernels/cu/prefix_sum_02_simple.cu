#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_simple(
    const unsigned int* a, // input n
          unsigned int* c, // output; n
    unsigned int n)
{
    int i = threadIdx.x;
    int glob_i = blockIdx.x * blockDim.x + i;
    __shared__ unsigned int locin[GROUP_SIZE];
    __shared__ unsigned int locout[GROUP_SIZE];

    if (glob_i < n) {
        locin[i] = a[glob_i];
    } else {
        locin[i] = 0;
    }
    locout[i] = 0;
    __syncthreads();

    // count prefix sum for WorkGroup
    int sz = 1;
    int mask = i + 1; //mask = [1; GROUP_SZ]
    while (sz <= GROUP_SIZE) { //maybe just <
        if (mask & sz) {
            mask -= sz;
            locout[i] += locin[mask];
            // printf("out[%d] += in[%d;%d) (=%u)\n", i, mask, mask + sz, locin[mask]);
        }
        __syncthreads();
        int l = i * sz;
        int r = (i + 1) * sz;
        if (r < GROUP_SIZE) { // +- ok for code divergence 
            locin[l] = locin[l] + locin[r]; // sum two blocks of size `sz`
        }
        __syncthreads();
        sz *= 2;
    }
    
    //load from local
    if (glob_i < n) {
        c[glob_i] = locout[i];;
    } 
}

namespace cuda {
void prefix_sum_simple(const gpu::WorkSize &workSize, 
    const gpu::gpu_mem_32u &a, unsigned int abase,
    gpu::gpu_mem_32u &c, unsigned int cbase,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_simple<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr() + abase, c.cuptr() + cbase, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
