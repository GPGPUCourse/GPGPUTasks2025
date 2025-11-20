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
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    unsigned int sorted_group_num = i / sorted_k;
    unsigned int sorted_group_index = i % sorted_k;
    unsigned int is_left_group = sorted_group_num % 2 == 0;

    if (i < n) {
        if (is_left_group) {
            unsigned int mirror_group_num = sorted_group_num + 1;

            unsigned int left_bound = sorted_k * mirror_group_num;
            unsigned int right_bound = min(sorted_k * (mirror_group_num + 1), n);

            while (left_bound < right_bound) {
                unsigned int mid = (left_bound + right_bound) / 2;
                if (input_data[mid] < input_data[i]) {
                    left_bound = mid + 1;
                } else {
                    right_bound = mid;
                }
            }
            output_data[left_bound - sorted_k + sorted_group_index] = input_data[i];
        } else {
            unsigned int mirror_group_num = sorted_group_num - 1;

            unsigned int left_bound = sorted_k * mirror_group_num;
            unsigned int right_bound = min(sorted_k * (mirror_group_num + 1), n);

            while (left_bound < right_bound) {
                unsigned int mid = (left_bound + right_bound) / 2;
                if (input_data[mid] <= input_data[i]) {
                    left_bound = mid + 1;
                } else {
                    right_bound = mid;
                }
            }
            output_data[right_bound + sorted_group_index] = input_data[i];
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
