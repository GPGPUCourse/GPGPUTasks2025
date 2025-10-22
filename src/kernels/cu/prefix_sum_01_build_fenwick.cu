#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define G_DEPTH 8 // 2^G_DEPTH = GROUP_SIZE
#define uint unsigned int

__global__ void prefix_sum_01_sum_build_fenwick(
    const uint* input,
          uint* output,
    uint n,
    uint d)
{
    __shared__ uint part[GROUP_SIZE];

    const uint bl_offset = (blockIdx.x * blockDim.x) << (d * 8);
    const uint g_idx = ((blockIdx.x * blockDim.x + threadIdx.x + 1) << (d * 8)) - 1;
    const uint idx = threadIdx.x;
    const uint l_size = min(GROUP_SIZE, n - bl_offset);

    if (g_idx < n) {
        part[idx] = input[g_idx];

        __syncthreads();

        #pragma unroll
        for (uint i = 1; i <= G_DEPTH; ++i) {
            const uint step = (1 << i);

            const uint j1 = (idx + 1) * step - 1;
            const uint j0 = j1 - (step >> 1);
            if (j1 < l_size) {
                part[j1] += part[j0];
            }

            __syncthreads();
        }

        output[g_idx] = part[idx];
    }
}

namespace cuda {
void prefix_sum_01_sum_build_fenwick(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input, gpu::gpu_mem_32u &output, uint n, uint d)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_sum_build_fenwick<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input.cuptr(), output.cuptr(), n, d);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
