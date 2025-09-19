#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void aplusb_matrix_good(const unsigned int* a,
                       const unsigned int* b,
                             unsigned int* c,
                             unsigned int  width,
                             unsigned int  height)
{
    // все три массива - линейно выложенные двумерные матрицы размера width (число столбиков) x height (число рядов)
    // при этом в памяти подряд идут элементы являющимися соседями в рамках одного ряда,
    // т.е. матрица выложена в памяти линейно ряд за рядом
    // т.е. если в матрице сделать шаг вправо или влево на одну ячейку - то в памяти мы шагнем на 4 байта
    // т.е. если в матрице сделать шаг вверх или вниз на одну ячейку - то в памяти мы шагнем на так называемый stride=width*4 байта

    // DONE реализуйте этот кернел - просуммируйте две матрицы так чтобы получить максимально ХОРОШУЮ производительность с точки зрения memory coalesced паттерна доступа

    // Ожидаем, что размер воркгруппы будет подгружать кэш линии целиком (напр. подходит 32x8, т.к. 32 * 4 = 128, или 256x1).
    const unsigned int column = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column >= width || row >= height) {
        return;
    }

    // Один воркайтем вычисляет одно число.
    const unsigned int idx = row * width + column;
    c[idx] = a[idx] + b[idx];
}

namespace cuda {
void aplusb_matrix_good(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, const gpu::gpu_mem_32u &b, gpu::gpu_mem_32u &c, unsigned int width, unsigned int height)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::aplusb_matrix_good<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), width, height);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
