#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

using u32 = unsigned int;

__global__ void GPUMerge(u32* in_buf, u32* out_buf, u32 n, u32 new_chunk_size) {
    u32 i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    u32 pos_in_new_chunk = i % new_chunk_size;
    u32 curr_chunk_size = new_chunk_size >> 1;

    bool is_left_half = (pos_in_new_chunk < curr_chunk_size);

    u32 curr_chunk_idx = i / curr_chunk_size;

    // [adjacent_chunk_l; adjacent_chunk_r)
    const u32 adjacent_chunk_l = (is_left_half * (curr_chunk_idx+1) * curr_chunk_size)
                               + (!is_left_half * (curr_chunk_idx-1) * curr_chunk_size);

    const u32 adjacent_chunk_r = min(adjacent_chunk_l + curr_chunk_size, n);
    u32 curr_value = in_buf[i];

    if (is_left_half && adjacent_chunk_l >= n) {
        out_buf[i] = curr_value;
        return;
    }
    // curassert(adjacent_chunk_l < adjacent_chunk_r, 321312312);
    // curassert(adjacent_chunk_l < n, 897210);
    // curassert(adjacent_chunk_r < n, 821893211);

    u32 new_chunk_idx = i / new_chunk_size;
    // printf("i=%d l=%d, r=%d, curr_chunk_idx=%d, new_chunk_idx=%d, is_left=%d\n", i,
    //         adjacent_chunk_l, adjacent_chunk_r, curr_chunk_idx, new_chunk_idx, is_left_half);

    // COMPUTE OFFSET IN ADJACENT CHUNK OF CURRENT SIZE
    u32 l = adjacent_chunk_l;
    u32 r = adjacent_chunk_r;

    u32 offset_in_current_chunk =  i % curr_chunk_size;
    u32 offset_in_ajacent_chunk = 0; // UPDATE VIA BINARY SEARCH


    while (l < r) {
        u32 mid = l + ((r - l) >> 1);
        u32 mid_value = in_buf[mid];

        bool go_right = is_left_half * (mid_value < curr_value) + !is_left_half * (mid_value <= curr_value);
        l = go_right * (mid+1) + !go_right * (l);
        r = go_right * (r) + !go_right * (mid);
    }

    offset_in_ajacent_chunk = r - adjacent_chunk_l;

    u32 output_res = (new_chunk_size * new_chunk_idx) + offset_in_current_chunk + offset_in_ajacent_chunk;
    // printf("%d > i(%d) curr_chunk_offset=%d aj_offset=%d output_res=%d\n", curr_value, i, offset_in_current_chunk, offset_in_ajacent_chunk, output_res);
    out_buf[output_res] = curr_value;
}

namespace cuda {

void MergeSort(gpu::gpu_mem_32u& b1, gpu::gpu_mem_32u& b2, int n) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    gpu::WorkSize ws(GROUP_SIZE, n);
    bool invert = false;
    for (std::size_t new_chunk_size = 2; (new_chunk_size >> 1) < n; new_chunk_size <<= 1) {
        ::GPUMerge<<<ws.cuGridSize(), ws.cuBlockSize(), 0, stream>>>
                  (b1.cuptr(), b2.cuptr(), n, new_chunk_size);
        CUDA_CHECK_KERNEL(stream);
        std::swap(b1, b2);
        invert = !invert;
    }

    if (invert) {
        std::swap(b1, b2);
    }

}

} // namespace cuda
