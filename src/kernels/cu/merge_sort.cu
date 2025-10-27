#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
    const unsigned int  sorted_k,
    const unsigned int  n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) return;

    const unsigned int num_block = index / sorted_k;
    const unsigned int start_block = num_block * sorted_k;
    const unsigned int end_block = min(num_block * sorted_k + sorted_k, n);

    const unsigned int start_block_o = (num_block & 1) ? start_block - sorted_k
        : start_block + sorted_k;
    const unsigned int end_block_o = min(start_block_o + sorted_k, n);

    if (end_block_o <= start_block_o) {
        output_data[index] = input_data[index];
        return;
    }

    int l = static_cast<int>(start_block_o) - 1;
    int r = static_cast<int>(end_block_o);

    const bool tp = start_block < start_block_o;
    const unsigned int my_val = input_data[index];
    while (r > l + 1) {
        const int m = (r + l) / 2;
        if (input_data[m] > my_val
            || (my_val == input_data[m] && tp)) {
            r = m;
        } else {
            l = m;
        }
    }

    const unsigned int count_before = r - start_block_o;
    const unsigned int start_new_block = min(start_block, start_block_o);

    output_data[start_new_block + index - start_block + count_before] = input_data[index];
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, unsigned int sorted_k, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
