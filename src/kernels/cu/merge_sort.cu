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
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n)
        return;

    size_t i = (index / (2 * sorted_k)) * (2 * sorted_k);
    size_t j = min(i + 2 * sorted_k, (unsigned long long)n);
    size_t mid = min(i + sorted_k, (unsigned long long)n);

    bool in_f = (index < mid);
    size_t idx = in_f ? index - i : index - mid;

    const unsigned int* f_block = input_data + i;
    const unsigned int* s_block = input_data + mid;
    size_t f = mid - i;
    size_t s = j - mid;

    unsigned int my_val = input_data[index];

    size_t l = 0, r = in_f ? s : f;
    while (l < r) {
        size_t m = (l + r) / 2;
        unsigned int cmp_val = in_f ? s_block[m] : f_block[m];
        if (cmp_val < my_val || (in_f && cmp_val == my_val))
            l = m + 1;
        else
            r = m;
    }

    size_t pos;
    if (in_f)
        pos = i + idx + l;
    else
        pos = i + l + idx;

    if (pos < n)
        output_data[pos] = my_val;
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
