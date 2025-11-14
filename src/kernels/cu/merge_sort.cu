#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "libbase/runtime_assert.h"

__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // if (i == 0) {
    //     printf("merge_sort kernel launched with sorted_k=%d, n=%d\n", sorted_k, n);
    //     printf("Current values: [%d, %d, %d, %d, %d, %d, %d, %d]\n", input_data[0], input_data[1], input_data[2], input_data[3], input_data[4], input_data[5], input_data[6], input_data[7]);
    // }
    
    if (i >= n) {
        return;
    }

    const int block = i / sorted_k;
    const int block_offset = block * sorted_k;
    const int idx_in_block = i % sorted_k;

    const int half_block_size = (sorted_k >> 1);
    const int idx_in_half_block = (idx_in_block >= half_block_size) ? (idx_in_block - half_block_size) : idx_in_block;

    int is_first_part = (idx_in_block < half_block_size);
    // curassert(is_first_part == 0 || is_first_part == 1, 123456789);

    int l = -1, r = half_block_size;
    const int cur_val = input_data[block_offset + idx_in_block];
    while (r - l > 1) {
        int m = (l + r) >> 1;
        const int idx_other = block_offset + half_block_size * is_first_part + m;
        if (idx_other >= n) {
            // printf("index %d index other %d out of bounds n %d\n", i, idx_other, n);
            r = m;
            continue;
        }

        // curassert(block_offset + idx_in_block < n && idx_other < n, 100432);
        const int other_val = input_data[idx_other];
        // printf("Comparing cur_val %d at input pos %d with other_val %d at input pos %d is first part %d\n", cur_val, block_offset + idx_in_block, other_val, idx_other, is_first_part);
        if (other_val < cur_val) {
            l = m;
        } else if (other_val > cur_val) {
            r = m;
        } else if (other_val == cur_val && is_first_part) {
            r = m;
        } else {
            l = m;
        }
    }

    const int output_pos = block_offset + idx_in_half_block + r;
    // printf("Inserting value %d from input pos %d to output pos %d index %d r %d\n", cur_val, block_offset + idx_in_block, output_pos, i, r);
    // curassert(output_pos < n, 987654321);
    output_data[output_pos] = input_data[i];
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
