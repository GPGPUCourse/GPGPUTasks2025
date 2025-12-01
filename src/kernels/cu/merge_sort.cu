#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/prim_gpu_cuda.h"

__global__ void merge_sort(
    const unsigned int* input_data_triIndex,
    const unsigned int* input_data_morton,
    const AABBGPU* input_data_aabb,
    const float3* input_data_centroid,
    unsigned int* output_data_triIndex,
    unsigned int* output_data_morton,
    AABBGPU* output_data_aabb,
    float3* output_data_centroid,
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
    const unsigned int cur_val = input_data_morton[block_offset + idx_in_block];
    while (r - l > 1) {
        int m = (l + r) >> 1;
        const int idx_other = block_offset + half_block_size * is_first_part + m;
        if (idx_other >= n) {
            // printf("index %d index other %d out of bounds n %d\n", i, idx_other, n);
            r = m;
            continue;
        }

        // curassert(block_offset + idx_in_block < n && idx_other < n, 100432);
        const unsigned int other_val = input_data_morton[idx_other];
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
    output_data_triIndex[output_pos] = input_data_triIndex[i];
    output_data_morton[output_pos] = input_data_morton[i];
    output_data_aabb[output_pos] = input_data_aabb[i];
    output_data_centroid[output_pos] = input_data_centroid[i];
}

namespace cuda {
void merge_sort(
    const gpu::WorkSize& workSize,
    const gpu::shared_device_buffer_typed<unsigned int>& input_data_triIndex,
    const gpu::shared_device_buffer_typed<unsigned int>& input_data_morton,
    const gpu::shared_device_buffer_typed<AABBGPU>& input_data_aabb,
    const gpu::shared_device_buffer_typed<float3>& input_data_centroid,
    gpu::shared_device_buffer_typed<unsigned int>& output_data_triIndex,
    gpu::shared_device_buffer_typed<unsigned int>& output_data_morton,
    gpu::shared_device_buffer_typed<AABBGPU>& output_data_aabb,
    gpu::shared_device_buffer_typed<float3>& output_data_centroid,
    int sorted_k,
    int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        input_data_triIndex.cuptr(), 
        input_data_morton.cuptr(), 
        input_data_aabb.cuptr(), 
        input_data_centroid.cuptr(), 
        output_data_triIndex.cuptr(), 
        output_data_morton.cuptr(), 
        output_data_aabb.cuptr(), 
        output_data_centroid.cuptr(), 
        sorted_k, 
        n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
