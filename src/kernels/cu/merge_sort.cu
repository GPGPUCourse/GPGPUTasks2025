#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__device__ void binary_search(
    const unsigned int* input_data,
    int start_low,
    int start_high,
    int key,
    bool second_part,
    int sorted_k,
    int* result)
{
    // [start_low .. start_high)
    if (key < input_data[start_low] || (!second_part && key == input_data[start_low])) {
        *result = start_low;
        return;
    }

    int left = start_low;
    int right = start_high;
    while (left + 1 < right) {
        int mid = (left + right) / 2;
        auto value = input_data[mid];
        if (value < key || (second_part && value == key)) {
            left = mid;
        } else {
            right = mid;
        }
    }
    *result = left + 1;
}

__global__ void merge_sort(
    const unsigned int* input_data,
    unsigned int* output_data,
    int sorted_k,
    int n)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        const unsigned int value = input_data[index];

        const int first_block_index = index / sorted_k;
        const int second_block_index = first_block_index + (first_block_index % 2 == 0 ? +1 : -1);
        const int local_index = index % sorted_k;

        const int left_block_index = (first_block_index % 2 == 0) ? first_block_index : second_block_index;

        int rindex = second_block_index * sorted_k;
        if (second_block_index * sorted_k < n) {
            binary_search(
                input_data,
                second_block_index * sorted_k,
                ((second_block_index + 1) * sorted_k < n ? (second_block_index + 1) * sorted_k : n),
                value,
                first_block_index % 2 != 0, sorted_k, &rindex);
        }
        const int second_block_search_index = rindex - second_block_index * sorted_k;

        const int insert_index = local_index + second_block_search_index;
        const int output_index = left_block_index * sorted_k + insert_index;
        output_data[output_index] = value;
    }
}

namespace cuda {
void merge_sort(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input_data, gpu::gpu_mem_32u& output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
