#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// find split for array a of size n and array b of size m at diagonal d (output pos)
__device__ int merge_path_find_i(const unsigned int* A, int a_len, const unsigned int* B, int b_len, int diag)
{
    int l = max(0, diag - b_len);
    int r = min(diag, a_len);
    while (l <= r) {
        int i = (l + r) >> 1;
        int j = diag - i;

        if ((i > 0) && (j < b_len) && (A[i - 1] > B[j])) {
            r = i - 1;
            continue;
        }

        if ((j > 0) && (i < a_len) && (B[j - 1] >= A[i])) {
            l = i + 1;
            continue;
        }

        return i;
    }
    return max(0, min(min(diag, a_len), diag - max(0, diag - b_len)));
}


// https://www.cs.ucdavis.edu/~amenta/f15/GPUmp.pdf
// merge parts size of sorted_k to parts size of 2 * sorted_k
__global__ void merge_sort(const unsigned int* input_data, unsigned int* output_data, int sorted_k, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    int group_size = sorted_k * 2;
    int offset = idx - (idx % group_size);
    int diag = idx - offset;

    int a_len = min(sorted_k, n - offset);
    int b_start = offset + a_len;
    int b_len = max(0, min(sorted_k, n - b_start));

    int total = a_len + b_len;
    if (diag >= total)
        return;

    const unsigned int* A = input_data + offset;
    const unsigned int* B = input_data + b_start;

    int i = merge_path_find_i(A, a_len, B, b_len, diag);
    int j = diag - i;

    unsigned int a = (i < a_len) ? A[i] : 0xFFFFFFFFu;
    unsigned int b = (j < b_len) ? B[j] : 0xFFFFFFFFu;

    output_data[offset + diag] = (a <= b) ? a : b;
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
