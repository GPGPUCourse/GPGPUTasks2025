#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include "copy_buffer.cu"

__global__ void merge_sort_kernel(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) {
        return;
    }
    
    const unsigned int key = input_data[idx];
    
    // bucket = pair of sorted arrays
    const int bucket_idx = idx / (2 * sorted_k);
    const int bucket_start = bucket_idx * 2 * sorted_k;
    
    // half intervals
    const int left_start = bucket_start;
    const int left_end = (left_start + sorted_k < n) ? (left_start + sorted_k) : n;

    const int right_start = left_end;
    const int right_end = (right_start + sorted_k < n) ? (right_start + sorted_k) : n;
    
    const bool is_in_right = (idx >= right_start);
    
    // corner case at the end of the array (no right part)
    if (right_start >= n) {
        output_data[idx] = key;
        return;
    }
    
    int pos_to_put = INT_MAX;
    
    if (is_in_right) {

        int l = left_start;
        int r = left_end;
        
        while (l < r) {
            int m = (l + r) / 2;
            if (input_data[m] <= key) { // <= for stability
                l = m + 1;
            } else {
                r = m;
            }
        }
        const int indice_in_left = l - left_start;
        const int indice_in_right = idx - right_start;

        pos_to_put = left_start + indice_in_left + indice_in_right;
    } else {
        int l = right_start;
        int r = right_end;
        
        while (l < r) {
            int m = (l + r) / 2;
            if (input_data[m] < key) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        const int indice_in_left = idx - left_start;
        const int indice_in_right = l - right_start;

        pos_to_put = left_start + indice_in_left + indice_in_right;
    }
    
    output_data[pos_to_put] = key;
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    
    gpu::gpu_mem_32u* current_input = &input_data;
    gpu::gpu_mem_32u* current_output = &output_data;
    
    int current_k = sorted_k;
    
    while (current_k < n) {
        merge_sort_kernel<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            current_input->cuptr(), 
            current_output->cuptr(), 
            current_k, 
            n
        );
        CUDA_CHECK_KERNEL(stream);
        
        std::swap(current_input, current_output);
        
        current_k *= 2;
    }
    
    if (current_input != &output_data) {
        ::copy_buffer<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream >> > (
            current_input->cuptr(), 
            output_data.cuptr(), 
            n);
        //cuda::copy_buffer(workSize, *current_input, output_data, n);
    }
}
} // namespace cuda
