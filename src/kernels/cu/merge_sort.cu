#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int run_start = blockIdx.x * (sorted_k << 1);
    if (run_start >= n) return;

    const unsigned int* A = input_data + run_start;
    const unsigned int* B = input_data + run_start + sorted_k;

    const int lenA = min(sorted_k, n - run_start);
    const int lenB = max(0, min(sorted_k, n - (run_start + sorted_k)));
    if (lenA <= 0) return;

    unsigned int* C = output_data + run_start;

    // нечего сортировать справа
    if (lenB == 0) {
        const int T = blockDim.x, t = threadIdx.x;
        const int chunk = (lenA + T - 1) / T;
        const int beg = t * chunk;
        const int end = min(lenA, beg + chunk);
        for (int i = beg; i < end; ++i) C[i] = A[i];
        return;
    }
    // сортировка
    if (threadIdx.x == 0) {
        int ia = 0, ib = 0, count = 0;

        while (ia < lenA && ib < lenB) {
            unsigned int va = A[ia];
            unsigned int vb = B[ib];

            if (va <= vb) {
                C[count++] = va;
                ia++;
            } else {
                C[count++] = vb;
                ib++;
            }
        }
        // для хвостов
        while (ia < lenA) {
            C[count++] = A[ia++];
        }
        while (ib < lenB) {
            C[count++] = B[ib++];
        }
    }
}
namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
