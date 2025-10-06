#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void mandelbrot(float* results,
                        unsigned int width, unsigned int height,
                        float fromX, float fromY,
                        float sizeX, float sizeY,
                        unsigned int iters, unsigned int isSmoothing)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO
}

namespace cuda {
void mandelbrot(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &results,
    unsigned int width, unsigned int height,
    float fromX, float fromY,
    float sizeX, float sizeY,
    unsigned int iters, unsigned int isSmoothing)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::mandelbrot<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(results.cuptr(), width, height, fromX, fromY, sizeX, sizeY, iters, isSmoothing);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
