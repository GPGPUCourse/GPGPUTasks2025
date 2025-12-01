#include <cfloat>
#include <device_launch_parameters.h>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <vector_types.h>

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

__global__ void build_bvh(
    const int n,
    BVHNodeGPU* outNodes,
    int* is_finished)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > n) {
        return;
    }

    auto& node = outNodes[i];
    auto leftChildIndex = node.leftChildIndex;
    auto rightChildIndex = node.rightChildIndex;

    if (leftChildIndex != UINT32_MAX && rightChildIndex != UINT32_MAX) {
        const BVHNodeGPU& left = outNodes[leftChildIndex];
        const BVHNodeGPU& right = outNodes[rightChildIndex];

        AABBGPU aabb;
        aabb.min_x = fmin(left.aabb.min_x, right.aabb.min_x);
        aabb.min_y = fmin(left.aabb.min_y, right.aabb.min_y);
        aabb.min_z = fmin(left.aabb.min_z, right.aabb.min_z);
        aabb.max_x = fmax(left.aabb.max_x, right.aabb.max_x);
        aabb.max_y = fmax(left.aabb.max_y, right.aabb.max_y);
        aabb.max_z = fmax(left.aabb.max_z, right.aabb.max_z);

        node.aabb = aabb;

        if (i == 0) {
            *is_finished = 1;
        }
    }
}

namespace cuda {
void build_bvh(const gpu::WorkSize& workSize,
    const int n,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes,
    gpu::shared_device_buffer_typed<int>& is_finished)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::build_bvh<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        n,
        nodes.cuptr(),
        is_finished.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
