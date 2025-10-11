#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

constexpr size_t BLOCK_DIM = GROUP_SIZE_X;

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int result_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int result_y_index = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int local_x_index = threadIdx.x;
    const unsigned int local_y_index = threadIdx.y;

    // Чтобы не перегружать доступ к данным, сначала посчитаем результат локально,
    // а затем его перепишем.
    __shared__ float local_result[GROUP_SIZE_X * GROUP_SIZE_Y];

    // Так как дальше подходим к задаче аддитивно, нужно занулить все целевые ячейки.
    local_result[local_y_index * GROUP_SIZE_X + local_x_index] = 0;

    // Выделяем память для подгрузки блоков.
    __shared__ float local_a_block[BLOCK_DIM * GROUP_SIZE_Y];
    __shared__ float local_b_block[BLOCK_DIM * GROUP_SIZE_X];

    // Будем подгружать пачку строк из `a`, и пачку столбцов из `b` блок за блоком,
    // вычислять для каждого блока все произведения и суммировать их в результат.
    for (size_t block_offset = 0; block_offset < k; block_offset += BLOCK_DIM) {
        // Вычислим индексы подгружаемых значений.
        const unsigned int load_a_x_index = local_x_index + block_offset;
        const unsigned int load_a_y_index = result_y_index;

        const unsigned int load_b_x_index = result_x_index;
        const unsigned int load_b_y_index = local_y_index + block_offset;

        // Подгрузим блок из строк `a`.
        if (load_a_x_index < k && load_a_y_index < h) {
            local_a_block[local_y_index * BLOCK_DIM + local_x_index] = a[load_a_y_index * k + load_a_x_index];
        }

        // Подгрузим блок из столбцов `b`.
        if (load_b_x_index < w && load_b_y_index < k) {
            local_b_block[local_y_index * BLOCK_DIM + local_x_index] = b[load_b_y_index * w + load_b_x_index];
        }

        __syncthreads();

        // Произведем вычисления на основе подгруженных блоков: каждый тред будет писать в свою ячейку.
        float result = 0.0f;
        for (size_t k_index = 0; k_index < BLOCK_DIM; ++k_index) {
            result += local_a_block[local_y_index * GROUP_SIZE_Y + k_index] * local_b_block[k_index * GROUP_SIZE_X + local_x_index];
        }
        local_result[local_y_index * GROUP_SIZE_X + local_x_index] += result;

        __syncthreads();
    }

    // Запишем результат в целевую таблицу.
    if (result_x_index < w && result_y_index < h) {
        c[result_y_index * w + result_x_index] = local_result[local_y_index * GROUP_SIZE_X + local_x_index];
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
