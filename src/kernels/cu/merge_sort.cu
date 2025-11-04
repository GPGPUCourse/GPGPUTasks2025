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
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t start = tid * 2ull * sorted_k;
    if (start >= (size_t)n)
        return;

    size_t mid = min(start + (size_t)sorted_k, (size_t)n);
    size_t end = min(start + 2ull * (size_t)sorted_k, (size_t)n);

    size_t i = start;
    size_t j = mid;
    size_t idx = start;

    while (i < mid && j < end) {
        if (input_data[i] <= input_data[j])
            output_data[idx++] = input_data[i++];
        else
            output_data[idx++] = input_data[j++];
    }
    while (i < mid)
        output_data[idx++] = input_data[i++];
    while (j < end)
        output_data[idx++] = input_data[j++];
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
