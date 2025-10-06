#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void mandelbrot(float* results,
    unsigned int width, unsigned int height,
    float fromX, float fromY,
    float sizeX, float sizeY,
    unsigned int iters, unsigned int isSmoothing)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) {
        return;
    }

    float cx = fromX + i * (sizeX / width);
    float cy = fromY + j * (sizeY / height);

    float x = 0.0f;
    float y = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    unsigned int iteration = 0;

    while (x2 + y2 <= 4.0f && iteration < iters) {
        y = 2.0f * x * y + cy;
        x = x2 - y2 + cx;
        x2 = x * x;
        y2 = y * y;
        iteration++;
    }

    unsigned int index = j * width + i;

    if (isSmoothing && iteration < iters) {
        float log_zn = logf(x2 + y2) / 2.0f;
        float nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
        results[index] = iteration + 1.0f - nu;
    } else {
        results[index] = (float)iteration;
    }
}

namespace cuda {
void mandelbrot(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& results,
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
