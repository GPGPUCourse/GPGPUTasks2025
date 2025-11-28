#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../shared_structs/camera_gpu_shared.h"
#include "../defines.h"
#include "helpers/rassert.cu"

#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"


__device__ __forceinline__ int common_prefix(const MortonCode* codes, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;

    MortonCode ci = codes[static_cast<size_t>(i)];
    MortonCode cj = codes[static_cast<size_t>(j)];

    if (ci == cj) {
        uint32_t di = static_cast<uint32_t>(i);
        uint32_t dj = static_cast<uint32_t>(j);
        uint32_t diff = di ^ dj;
        return 32 + __clz(diff);
    }
    uint32_t diff = ci ^ cj;
    return __clz(diff);
}

__global__ void make_lbvh(
    const MortonCode* codes,
    const unsigned int* leaf_indices,
    const float* vertices,
    const unsigned int* faces,
    unsigned int nfaces,
    BVHNodeGPU* bvh_nodes,
    int* indices_up
)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nfaces) return;

    uint3 f = loadFace(faces, leaf_indices[index]);
    float3 a = loadVertex(vertices, f.x);
    float3 b = loadVertex(vertices, f.y);
    float3 c = loadVertex(vertices, f.z);

    bvh_nodes[nfaces - 1 + index].aabb = {
        min(min(b.x, c.x), a.x),
        min(min(b.y, c.y), a.y),
        min(min(b.z, c.z), a.z),

        max(max(b.x, c.x), a.x),
        max(max(b.y, c.y), a.y),
        max(max(b.z, c.z), a.z),
    };

    bvh_nodes[nfaces - 1 + index].leftChildIndex = INT_MAX;
    bvh_nodes[nfaces - 1 + index].rightChildIndex = INT_MAX;

    if (index + 1 >= nfaces) return;

    int cnt_r = common_prefix(codes, nfaces, index + 1, index);
    int cnt_l = -1;
    if (index != 0) {
        cnt_l = common_prefix(codes, nfaces, index - 1, index);
    }
    int mn_cnt = min(cnt_l, cnt_r);

    int l = index, r = nfaces;
    if (cnt_l > cnt_r) {
        l = -1, r = index;
    }

    while (l + 1 < r) {
        int m = l + (r - l) / 2;
        int cnt = common_prefix(codes, nfaces, m, index);
        if (cnt > mn_cnt) {
            if (cnt_l > cnt_r) {
                r = m;
            } else {
                l = m;
            }
        } else {
            if (cnt_l > cnt_r) {
                l = m;
            } else {
                r = m;
            }
        }
    }

    int l_split, r_split;
    int split_cnt;
    int left_, right_;
    if (cnt_l > cnt_r) {
        l_split = r;
        r_split = index;
        split_cnt = common_prefix(codes, nfaces, l_split, index);
        left_ = l_split;
        right_ = r_split;
    } else {
        l_split = index;
        r_split = l;
        split_cnt = common_prefix(codes, nfaces, r_split, index);
        left_ = l_split;
        right_ = r_split;
    }

    while (l_split + 1 < r_split) {
        int m = l_split + (r_split - l_split) / 2;
        int cnt = common_prefix(codes, nfaces, m, index);
        if (cnt > split_cnt) {
            if (cnt_l > cnt_r) {
                r_split = m;
            } else {
                l_split = m;
            }
        } else {
            if (cnt_l > cnt_r) {
                l_split = m;
            } else {
                r_split = m;
            }
        }
    }

    bvh_nodes[index].aabb = {FLT_MAX, FLT_MAX, FLT_MAX,
        -FLT_MAX, -FLT_MAX, -FLT_MAX};

    if (l_split == left_) {
        bvh_nodes[index].leftChildIndex = l_split + nfaces - 1;
    } else {
        bvh_nodes[index].leftChildIndex = l_split;
        indices_up[l_split] = index;
    }
    if (l_split + 1 == right_) {
        bvh_nodes[index].rightChildIndex = l_split + nfaces;
    } else {
        bvh_nodes[index].rightChildIndex = l_split + 1;
        indices_up[l_split + 1] = index;
    }

    if (index == 0) {
        indices_up[0] = -1;
    }
}


namespace cuda {
    void make_lbvh(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &codes,
        const gpu::gpu_mem_32u &leaf_indices,
        const gpu::gpu_mem_32f &vertices,
        const gpu::gpu_mem_32u &faces,
        unsigned int nfaces,
        gpu::shared_device_buffer_typed<BVHNodeGPU> &bvh_nodes,
        gpu::gpu_mem_32i &indices_up)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::make_lbvh<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        codes.cuptr(),
        leaf_indices.cuptr(),
        vertices.cuptr(),
        faces.cuptr(),
        nfaces,
        bvh_nodes.cuptr(),
        indices_up.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
