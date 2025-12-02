#include <cfloat>
#include <device_launch_parameters.h>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <vector_types.h>

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

__device__ int common_prefix(const MortonCode* codes, int N, int i, int j)
{
    if (j < 0 || j >= N)
        return -1;

    MortonCode ci = codes[i];
    MortonCode cj = codes[j];

    if (ci == cj) {
        unsigned int di = *(unsigned int*)(&i);
        unsigned int dj = *(unsigned int*)(&j);
        unsigned int diff = di ^ dj;
        return 32 + __clz(diff);
    } else {
        unsigned int diff = ci ^ cj;
        return __clz(diff);
    }
}

// Determine range [first, last] of primitives covered by internal node i
__device__ void determine_range(const MortonCode* codes, int N, int i, int& outFirst, int& outLast)
{
    int cpL = common_prefix(codes, N, i, i - 1);
    int cpR = common_prefix(codes, N, i, i + 1);

    // Direction of the range
    int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    int deltaMin = common_prefix(codes, N, i, i - d);
    int lmax = 2;

    while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    // Binary search to find exact range length
    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    outFirst = min(i, j);
    outLast = max(i, j);
}

// Find split position inside range [first, last] using the same
// prefix metric as determine_range (code + index tie-break)
__device__ int find_split(const MortonCode* codes, int N, int first, int last)
{
    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last)
        return first;

    // Prefix between first and last (уже с учётом индекса, если коды равны)
    int commonPrefix = common_prefix(codes, N, first, last);

    int split = first;
    int step = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = common_prefix(codes, N, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

__global__ void pre_build_bvh(
    const int n,
    const unsigned int* data_triIndex,
    const MortonCode* data_morton,
    const AABBGPU* data_aabb,
    BVHNodeGPU* outNodes,
    int* parentIndices)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 2 * n - 2) {
        return;
    }

    /*
     // 5) Initialize leaf nodes [N-1 .. 2*N-2]
    constexpr uint32_t INVALID = std::numeric_limits<uint32_t>::max();

    for (size_t i = 0; i < N; ++i) {
        size_t leafIndex = (N - 1) + i;
        BVHNodeGPU& leaf = outNodes[leafIndex];

        leaf.aabb = prims[i].aabb;
        leaf.leftChildIndex = INVALID;
        leaf.rightChildIndex = INVALID;
    }

    // 6) Build internal nodes [0 .. N-2]
    for (int i = 0; i < static_cast<int>(N) - 1; ++i) {
        int first, last;
        determine_range(sortedCodes, static_cast<int>(N), i, first, last);
        int split = find_split(sortedCodes, first, last);

        // Left child
        int leftIndex;
        if (split == first) {
            // Range [first, split] has one primitive -> leaf
            leftIndex = static_cast<int>((N - 1) + split);
        } else {
            // Internal node
            leftIndex = split;
        }

        // Right child
        int rightIndex;
        if (split + 1 == last) {
            // Range [split+1, last] has one primitive -> leaf
            rightIndex = static_cast<int>((N - 1) + split + 1);
        } else {
            // Internal node
            rightIndex = split + 1;
        }

        BVHNodeGPU& node = outNodes[static_cast<size_t>(i)];
        node.leftChildIndex = static_cast<uint32_t>(leftIndex);
        node.rightChildIndex = static_cast<uint32_t>(rightIndex);
    }
    */

    if (i > n - 2) {
        BVHNodeGPU& leaf = outNodes[i];

        leaf.aabb = data_aabb[i - n + 1];
        leaf.leftChildIndex = UINT32_MAX;
        leaf.rightChildIndex = UINT32_MAX;
    } else {
        int first, last;
        determine_range(data_morton, n, i, first, last);
        int split = find_split(data_morton, n, first, last);

        // Left child
        int leftIndex;
        if (split == first) {
            // Range [first, split] has one primitive -> leaf
            leftIndex = (n - 1) + split;
        } else {
            // Internal node
            leftIndex = split;
        }

        // Right child
        int rightIndex;
        if (split + 1 == last) {
            // Range [split+1, last] has one primitive -> leaf
            rightIndex = (n - 1) + split + 1;
        } else {
            // Internal node
            rightIndex = split + 1;
        }

        BVHNodeGPU& node = outNodes[i];
        node.leftChildIndex = (unsigned int)leftIndex;
        node.rightChildIndex = (unsigned int)rightIndex;

        parentIndices[leftIndex] = i;
        parentIndices[rightIndex] = i;
    }

    if (i == 0) {
        parentIndices[0] = -1;
    }
}

namespace cuda {
void pre_build_bvh(const gpu::WorkSize& workSize,
    const int n,
    gpu::shared_device_buffer_typed<unsigned int>& data_triIndex,
    gpu::shared_device_buffer_typed<MortonCode>& data_morton,
    gpu::shared_device_buffer_typed<AABBGPU>& data_aabb,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& outNodes,
    gpu::shared_device_buffer_typed<int>& parentIndices)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::pre_build_bvh<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        n,
        data_triIndex.cuptr(),
        data_morton.cuptr(),
        data_aabb.cuptr(),
        outNodes.cuptr(),
        parentIndices.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
