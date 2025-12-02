#include <cfloat>
#include <device_launch_parameters.h>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <vector_types.h>

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

__global__ void fill_zeros(int* data, const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = 0;
    }
}

__global__ void build_bvh(
    const int nLeaves,
    BVHNodeGPU* nodes,
    const int* parentIndices,
    int* atomicCounters)
{
    const int leafIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (leafIdx >= nLeaves) {
        return;
    }

    int nodeIdx = (nLeaves - 1) + leafIdx;

    while (true) {
        int parentIdx = __ldg(&parentIndices[nodeIdx]);

        if (parentIdx < 0) {
            break;
        }

        int count = atomicAdd(&atomicCounters[parentIdx], 1);

        if (count == 0) {
            break;
        }

        BVHNodeGPU& parent = nodes[parentIdx];
        const BVHNodeGPU& left = nodes[parent.leftChildIndex];
        const BVHNodeGPU& right = nodes[parent.rightChildIndex];

        AABBGPU aabb;
        aabb.min_x = fminf(left.aabb.min_x, right.aabb.min_x);
        aabb.min_y = fminf(left.aabb.min_y, right.aabb.min_y);
        aabb.min_z = fminf(left.aabb.min_z, right.aabb.min_z);
        aabb.max_x = fmaxf(left.aabb.max_x, right.aabb.max_x);
        aabb.max_y = fmaxf(left.aabb.max_y, right.aabb.max_y);
        aabb.max_z = fmaxf(left.aabb.max_z, right.aabb.max_z);

        parent.aabb = aabb;

        nodeIdx = parentIdx;
    }
}

namespace cuda {

void fill_zeros(const gpu::WorkSize& workSize,
    gpu::shared_device_buffer_typed<int>& data,
    const int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124314, context.type());
    cudaStream_t stream = context.cudaStream();
    ::fill_zeros<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        data.cuptr(),
        n
    );
    CUDA_CHECK_KERNEL(stream);
}

void build_bvh(const gpu::WorkSize& workSize,
    const int nLeaves,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes,
    gpu::shared_device_buffer_typed<int>& parentIndices,
    gpu::shared_device_buffer_typed<int>& atomicCounters)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124313, context.type());
    cudaStream_t stream = context.cudaStream();
    ::build_bvh<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        nLeaves,
        nodes.cuptr(),
        parentIndices.cuptr(),
        atomicCounters.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda
