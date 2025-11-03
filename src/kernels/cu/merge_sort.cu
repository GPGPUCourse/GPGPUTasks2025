#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void merge_sort(
    const unsigned int* input_data,
    unsigned int* output_data,
    int sorted_k,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int block_idx = idx / sorted_k;
    const int block_elem_idx = idx % sorted_k;

    const int block2_idx = block_idx + (block_idx % 2 == 0 ? 1 : -1);

    const unsigned int elem = __ldg(input_data + idx);
    if (block2_idx < 0 || block2_idx * sorted_k >= n) {
        output_data[idx] = elem;
        return;
    }

    unsigned int l = block2_idx * sorted_k - 1;
    unsigned int r =  min((block2_idx + 1) * sorted_k, n);
    const unsigned int is_right = (block_idx > block2_idx);
    while (r - l > 1) {
        const unsigned int m = (l + r) / 2;
        const unsigned int a = __ldg(input_data + m);
        // '<' if left and '<=' if right
        (a < elem + is_right ? l : r) = m;
    }
    curassert(l + 1 == r, 4200042000);

    r = (r / sorted_k == block2_idx ? r % sorted_k : sorted_k);

    const unsigned int out_idx = min(block_idx, block2_idx) * sorted_k + block_elem_idx + r;
    // printf("%d -> %d (global_offset=%d,block_offset=%d,bin_search_offset=%d)\n", idx, out_idx, min(block_idx, block2_idx) * sorted_k, block_elem_idx, r);
    output_data[out_idx] = elem;
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