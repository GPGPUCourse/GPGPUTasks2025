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

    const float THRESHOLD = 256.0f;
    const float THRESHOLD_SQUARE = THRESHOLD * THRESHOLD;
    if (i >= width || j >= height) return;
    const float x_begin = fromX + (i + 0.5f) * sizeX / width;
    const float y_begin = fromY + (j + 0.5f) * sizeY / height;
    unsigned int iter = 0;
    float current_x = x_begin;
    float current_y = y_begin;
    for (; iter < iters; ++iter) {
        float previous_x = current_x;
        current_x = current_x * current_x - current_y * current_y + x_begin;
        current_y = 2.0f * previous_x * current_y + y_begin;
        if ((current_x * current_x + current_y * current_y) > THRESHOLD_SQUARE) {
            break;
        }
    }
    float result = iter;
    if (isSmoothing && iter != iters) {
        result -= log(log(sqrt(current_x * current_x + current_y * current_y)) / (log(THRESHOLD) * log(2.0f)));
    }
    result *= 1.0f/iters;
    results[i + j * width] = result;
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
