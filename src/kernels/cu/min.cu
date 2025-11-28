#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void min_array(
    const float* a,
    const float* b,
    const float* c,
    const unsigned int n,
    float* out_a,
    float* out_b,
    float* out_c)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float local_min_a[GROUP_SIZE];
    __shared__ float local_min_b[GROUP_SIZE];
    __shared__ float local_min_c[GROUP_SIZE];

    const unsigned int step = (n + 1) / 2;

    float mn_a = FLT_MAX;
    float mn_b = FLT_MAX;
    float mn_c = FLT_MAX;
    if (index < step) {
        for (unsigned int i = 0; i < 2; i++) {
            const unsigned int id = i * step + index;
            if (id < n) {
                mn_a = min(mn_a, a[id]);
                mn_b = min(mn_b, b[id]);
                mn_c = min(mn_c, c[id]);
            }
        }
    }
    local_min_a[threadIdx.x] = mn_a;
    local_min_b[threadIdx.x] = mn_b;
    local_min_c[threadIdx.x] = mn_c;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (const auto el: local_min_a) {
            mn_a = min(mn_a, el);
        }
        for (const auto el: local_min_b) {
            mn_b = min(mn_b, el);
        }
        for (const auto el: local_min_c) {
            mn_c = min(mn_c, el);
        }
        out_a[blockIdx.x] = mn_a;
        out_b[blockIdx.x] = mn_b;
        out_c[blockIdx.x] = mn_c;
    }
}

namespace cuda {
void min_array(const gpu::WorkSize &workSize, const gpu::gpu_mem_32f &a,
    const gpu::gpu_mem_32f &b, const gpu::gpu_mem_32f &c,
    unsigned int n, gpu::gpu_mem_32f &out_a, gpu::gpu_mem_32f &out_b, gpu::gpu_mem_32f &out_c)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::min_array<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(),
        c.cuptr(), n, out_a.cuptr(), out_b.cuptr(), out_c.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
