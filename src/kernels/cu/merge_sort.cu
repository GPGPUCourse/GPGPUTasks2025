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
    if (i >= n) return;

    const int block = i / sorted_k;
    const int pair = block / 2;

    const int left_block = pair * 2;
    const int right_block = left_block + 1;

    const int l_s = left_block * sorted_k;
    const int l_e = min(l_s + sorted_k, n);
    const int r_s = right_block * sorted_k;
    const int r_e = min(r_s + sorted_k, n);

    int search_l, search_r;
    int self_s;

    if (block % 2 == 0) {
        search_l = r_s;
        search_r = r_e;
        self_s = l_s;
    } else {
        search_l = l_s;
        search_r = l_e;
        self_s = r_s;
    }

    const unsigned int value = input_data[i];

    int l = search_l, r = search_r;
    while (l < r) {
        int m = (l + r) / 2;
        if (input_data[m] < value || (block % 2 == 1 && input_data[m] == value))
            l = m + 1;
        else
            r = m;
    }

    const int rank = (i - self_s) + (l - search_l);
    const int out = pair * 2 * sorted_k + rank;

    output_data[out] = value;
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
