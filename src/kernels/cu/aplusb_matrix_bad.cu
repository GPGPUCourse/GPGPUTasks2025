#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define CACHELINE_SIZE_UINT 32  // (128 / sizeof(unsigned int))

__global__ void aplusb_matrix_bad(const unsigned int* a,
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

    // DONE реализуйте этот кернел - просуммируйте две матрицы так чтобы получить максимально ПЛОХУЮ производительность с точки зрения memory coalesced паттерна доступа

    // Для сравнимости с хорошим вариантом пришлось реализовать схожий подход -- один ворк-айтем = одно суммирование
    // (и сразу воркгруппу 32x8). Для того чтобы достигнуть очень плохой производительности, будем делать так, чтобы
    // каждый тред читал число из своей кэшлинии, но идейно, как и в хорошем кернеле, будем просить ворк-айтем считать
    // только одно значение (возможно, еще и какой-нибудь аналог false sharing между ворк-группами получим, т.к. разные
    // воркгруппы будут пытаться изменить одну кэш-линию).
    //
    // Все строки заведомо в различных кэш-линиях, значит будем двигать
    // только индекс колонки. Выбирать элементы будем "расческой".
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Тред -- из какого блока в строке берем значение.
    const unsigned int col_block_idx = threadIdx.x;
    // Ширина одного блока: одна воркгруппа должна посчитать по одному значению
    // в каждом блоке строки, поэтому делим ширину на кол-во воркайтемов на строку.
    const unsigned int col_block_width = width / blockDim.x;
    // Отступ внутри каждого блока равен индексу самого блока.
    const unsigned int col_block_offset = blockIdx.x;

    const unsigned int idx = row * width + col_block_width * col_block_idx + col_block_offset;
    c[idx] = a[idx] + b[idx];
}

namespace cuda {
void aplusb_matrix_bad(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, const gpu::gpu_mem_32u &b, gpu::gpu_mem_32u &c, unsigned int width, unsigned int height)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::aplusb_matrix_bad<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), width, height);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
