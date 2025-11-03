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
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int block_size = 2 * sorted_k;
    const unsigned int block_id = i / block_size;
    const unsigned int block_start = block_id * block_size;
    const unsigned int block_mid = block_start + sorted_k;

    if (block_mid >= n) {
        output_data[i] = input_data[i];
        return;
    }

    const unsigned int block_end = min(block_start + block_size, n);
    const bool is_right = i >= block_mid;

    unsigned int lb = (is_right ? block_start : block_mid) - 1;
    unsigned int rb = (is_right ? block_mid : block_end);

    unsigned int value = input_data[i];

    while (rb - lb > 1) {
        unsigned int m = (lb + rb) / 2;
        if ((is_right && input_data[m] > value) || (!is_right && input_data[m] >= value)) {
            rb = m;
        } else {
            lb = m;
        }
    }

    unsigned int i_out = block_start + (is_right ?
        (i - block_mid) + (rb - block_start) :
        (i - block_start) + (rb - block_mid));

    output_data[i_out] = value;
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
