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

    if (i >= width || j >= height) return;
    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;
    float x = x0;
    float y = y0;

    unsigned int iter = 0;
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    // Основной цикл
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;

        if ((x * x + y * y) > threshold2) {
            break;
        }
    }

    // Нормализация результата
    float result = (float)iter;
    if (isSmoothing && iter != iters) {
        result = result - logf(logf(sqrtf(x * x + y * y)) / logf(threshold)) / logf(2.0f);
    }

    result = result / (float)iters;

    // Записываем в выходной буфер
    results[j * width + i] = result;
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
