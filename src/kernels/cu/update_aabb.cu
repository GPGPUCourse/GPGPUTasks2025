#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void update_aabb(
    BVHNodeGPU* nodes,
    const int* indices_up,
    const unsigned int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index + 1 >= n) return;

    do {
        if (nodes[index].leftChildIndex < n + n - 1) {
            const auto child = nodes[nodes[index].leftChildIndex].aabb;
            const auto tmp = nodes[index].aabb;

            nodes[index].aabb = {
                min(child.min_x, tmp.min_x),
                min(child.min_y, tmp.min_y),
                min(child.min_z, tmp.min_z),
                max(child.max_x, tmp.max_x),
                max(child.max_y, tmp.max_y),
                max(child.max_z, tmp.max_z),
            };
        }
        if (nodes[index].rightChildIndex < n + n - 1) {
            const auto child = nodes[nodes[index].rightChildIndex].aabb;
            const auto tmp = nodes[index].aabb;

            nodes[index].aabb = {
                min(child.min_x, tmp.min_x),
                min(child.min_y, tmp.min_y),
                min(child.min_z, tmp.min_z),
                max(child.max_x, tmp.max_x),
                max(child.max_y, tmp.max_y),
                max(child.max_z, tmp.max_z),
            };
        }
        index = indices_up[index];
    } while (index >= 0 && index + 1 < n);
}

namespace cuda {
    void update_aabb(const gpu::WorkSize &workSize,
        gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes,
        const gpu::gpu_mem_32i &indices_up,
        const unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::update_aabb<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(nodes.cuptr(),
        indices_up.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
