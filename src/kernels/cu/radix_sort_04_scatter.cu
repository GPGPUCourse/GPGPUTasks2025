#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_04_scatter(
    const unsigned int* input,
    const unsigned int* indicies,
    unsigned int* output,
    unsigned int pow2,
    unsigned int n)
{
    // index in input array
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    // index in output array
    if (index < n) {
        int tindex = -1;
        const int value = input[index];
        if ((value & (1 << pow2)) == 0) {
            // first segment
            tindex = indicies[index] - 1;
        } else {
            // second segment
            tindex = indicies[n - 1] + (index - indicies[index]);
        }

        output[tindex] = value;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& buffer1, const gpu::gpu_mem_32u& buffer2, gpu::gpu_mem_32u& buffer3, unsigned int a1, unsigned int a2)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(buffer1.cuptr(), buffer2.cuptr(), buffer3.cuptr(), a1, a2);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
