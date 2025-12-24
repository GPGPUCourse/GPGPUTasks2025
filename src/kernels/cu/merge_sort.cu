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

    if (i >= n) {
        return;
    }

    unsigned int array_pos = i >> sorted_k;
    unsigned int friend_array_pos = array_pos ^ 1;

    unsigned int arr_len = 1 << sorted_k;
    unsigned int array_start = array_pos * arr_len;
    unsigned int friend_start = friend_array_pos * arr_len;

    // Array is sorted or tail has no pair
    if (arr_len == n || friend_start >= n) {
        output_data[i] = input_data[i];
        return;
    }

    unsigned int friend_end = 
        friend_start + arr_len < n ? friend_start + arr_len : n;

    auto cmp = 
        friend_array_pos % 2 == 0 ? [](int a, int b){ return a < b; } : [](int a, int b){ return a <= b; };
    
    int l = friend_start - 1;
    int r = friend_end;

    while (r - l > 1) {
        int m = l + (r - l) / 2;
        
        if (cmp(input_data[i], input_data[m])) {
            r = m;
        } else {
            l = m;
        }
    }

    unsigned int i_relative = i % arr_len;
    unsigned int r_relative = r - friend_start;
    unsigned int new_arr_start = friend_array_pos % 2 == 0 ? friend_start : array_start;

    output_data[new_arr_start + i_relative + r_relative] = input_data[i];
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
