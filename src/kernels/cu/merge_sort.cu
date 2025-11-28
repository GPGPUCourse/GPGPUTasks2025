#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "../shared_structs/morton_code_gpu_shared.h"

__global__ void merge_sort(
    const unsigned int* indices,
    const MortonCode*   morton_codes,
          MortonCode*   out_morton_codes,
          unsigned int* out_indices,
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
        out_morton_codes[index] = morton_codes[index];
        out_indices[index] = indices[index];
        return;
    }

    int l = static_cast<int>(start_block_o) - 1;
    int r = static_cast<int>(end_block_o);

    const bool tp = start_block < start_block_o;
    const unsigned int my_val = morton_codes[index];
    while (r > l + 1) {
        const int m = (r + l) / 2;
        if (morton_codes[m] > my_val
            || (my_val == morton_codes[m] && tp)) {
            r = m;
        } else {
            l = m;
        }
    }

    const unsigned int count_before = r - start_block_o;
    const unsigned int start_new_block = min(start_block, start_block_o);

    out_morton_codes[start_new_block + index - start_block + count_before] = morton_codes[index];
    out_indices[start_new_block + index - start_block + count_before] = indices[index];
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &indices, const gpu::gpu_mem_32u &morton_codes,
            gpu::gpu_mem_32u& out_morton_codes, gpu::gpu_mem_32u &out_indices,
            unsigned int sorted_k, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(indices.cuptr(),
        morton_codes.cuptr(), out_morton_codes.cuptr(), out_indices.cuptr(),
        sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
