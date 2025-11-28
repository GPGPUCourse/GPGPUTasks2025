#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void max_array(
    const float* a,
    const float* b,
    const float* c,
    const unsigned int n,
    float* out_a,
    float* out_b,
    float* out_c)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float local_max_a[GROUP_SIZE];
    __shared__ float local_max_b[GROUP_SIZE];
    __shared__ float local_max_c[GROUP_SIZE];

    const unsigned int step = (n + 1) / 2;

    float mx_a = -FLT_MAX;
    float mx_b = -FLT_MAX;
    float mx_c = -FLT_MAX;
    if (index < step) {
        for (unsigned int i = 0; i < 2; i++) {
            const unsigned int id = i * step + index;
            if (id < n) {
                mx_a = max(mx_a, a[id]);
                mx_b = max(mx_b, b[id]);
                mx_c = max(mx_c, c[id]);
            }
        }
    }
    local_max_a[threadIdx.x] = mx_a;
    local_max_b[threadIdx.x] = mx_b;
    local_max_c[threadIdx.x] = mx_c;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (const auto el: local_max_a) {
            mx_a = max(mx_a, el);
        }
        for (const auto el: local_max_b) {
            mx_b = max(mx_b, el);
        }
        for (const auto el: local_max_c) {
            mx_c = max(mx_c, el);
        }
        out_a[blockIdx.x] = mx_a;
        out_b[blockIdx.x] = mx_b;
        out_c[blockIdx.x] = mx_c;
    }
}

namespace cuda {
void max_array(const gpu::WorkSize &workSize, const gpu::gpu_mem_32f &a,
    const gpu::gpu_mem_32f &b, const gpu::gpu_mem_32f &c,
    unsigned int n, gpu::gpu_mem_32f &out_a, gpu::gpu_mem_32f &out_b, gpu::gpu_mem_32f &out_c)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::max_array<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(),
        c.cuptr(), n, out_a.cuptr(), out_b.cuptr(), out_c.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
