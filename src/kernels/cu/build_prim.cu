#include <cfloat>
#include <device_launch_parameters.h>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <vector_types.h>

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/prim_gpu_cuda.h"

/*
    const size_t N = faces.size();
    outNodes.clear();
    outLeafTriIndices.clear();

    if (N == 0) {
        return;
    }

    // Special case: single triangle -> single leaf/root
    if (N == 1) {
        outNodes.resize(1);
        outLeafTriIndices.resize(1);

        const point3u& f = faces[0];
        const point3f& v0 = vertices[f.x];
        const point3f& v1 = vertices[f.y];
        const point3f& v2 = vertices[f.z];

        AABBGPU aabb;
        aabb.min_x = std::min({v0.x, v1.x, v2.x});
        aabb.min_y = std::min({v0.y, v1.y, v2.y});
        aabb.min_z = std::min({v0.z, v1.z, v2.z});
        aabb.max_x = std::max({v0.x, v1.x, v2.x});
        aabb.max_y = std::max({v0.y, v1.y, v2.y});
        aabb.max_z = std::max({v0.z, v1.z, v2.z});

        BVHNodeGPU& node = outNodes[0];
        node.aabb = aabb;
        // Leaf: no children (user can detect leaf via index and N)
        node.leftChildIndex  = std::numeric_limits<GPUC_UINT>::max();
        node.rightChildIndex = std::numeric_limits<GPUC_UINT>::max();

        outLeafTriIndices[0] = 0;
        return;
    }

    // Per-triangle info
    struct Prim {
        uint32_t triIndex;
        uint32_t morton;
        AABBGPU  aabb;
        point3f  centroid;
    };

    std::vector<Prim> prims(N);

    // 1) Compute per-triangle AABB and centroids
    point3f cMin{+std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity()};
    point3f cMax{-std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};

    for (size_t i = 0; i < N; ++i) {
        const point3u& f = faces[i];
        const point3f& v0 = vertices[f.x];
        const point3f& v1 = vertices[f.y];
        const point3f& v2 = vertices[f.z];

        // Triangle AABB
        AABBGPU aabb;
        aabb.min_x = std::min({v0.x, v1.x, v2.x});
        aabb.min_y = std::min({v0.y, v1.y, v2.y});
        aabb.min_z = std::min({v0.z, v1.z, v2.z});
        aabb.max_x = std::max({v0.x, v1.x, v2.x});
        aabb.max_y = std::max({v0.y, v1.y, v2.y});
        aabb.max_z = std::max({v0.z, v1.z, v2.z});

        // Centroid
        point3f c;
        c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
        c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
        c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

        prims[i].triIndex = static_cast<uint32_t>(i);
        prims[i].aabb     = aabb;
        prims[i].centroid = c;

        // Update centroid bounds
        cMin.x = std::min(cMin.x, c.x);
        cMin.y = std::min(cMin.y, c.y);
        cMin.z = std::min(cMin.z, c.z);
        cMax.x = std::max(cMax.x, c.x);
        cMax.y = std::max(cMax.y, c.y);
        cMax.z = std::max(cMax.z, c.z);
    }

    // 2) Compute Morton codes for centroids (normalized to [0,1]^3)
    const float eps = 1e-9f;
    const float dx = std::max(cMax.x - cMin.x, eps);
    const float dy = std::max(cMax.y - cMin.y, eps);
    const float dz = std::max(cMax.z - cMin.z, eps);

    for (size_t i = 0; i < N; ++i) {
        const point3f& c = prims[i].centroid;
        float nx = (c.x - cMin.x) / dx;
        float ny = (c.y - cMin.y) / dy;
        float nz = (c.z - cMin.z) / dz;

        // Clamp to [0,1]
        nx = std::min(std::max(nx, 0.0f), 1.0f);
        ny = std::min(std::max(ny, 0.0f), 1.0f);
        nz = std::min(std::max(nz, 0.0f), 1.0f);

        prims[i].morton = morton3D(nx, ny, nz);
    }
    */

__device__ void atomicMinFloat(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    float& assumed_val = *(float*)&assumed;

    do {
        assumed = old;
        if (assumed_val <= val) break;
        old = atomicCAS(address_as_i, assumed, *(int*)&val);
    } while (assumed != old);
}

__device__ void atomicMaxFloat(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    float& assumed_val = *(float*)&assumed;

    do {
        assumed = old;
        if (assumed_val >= val) break;
        old = atomicCAS(address_as_i, assumed, *(int*)&val);
    } while (assumed != old);
}

__global__ void build_prim(
    const float* vertices,
    const unsigned int* faces,
    const int nVertices,
    const int nFaces,
    unsigned int* data_triIndex,
    unsigned int* data_morton,
    AABBGPU* data_aabb,
    float3* data_centroid,
    float3* cMin,
    float3* cMax)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nFaces) {
        return;
    }

    /*
    std::vector<Prim> prims(N);

    // 1) Compute per-triangle AABB and centroids
    point3f cMin{+std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity()};
    point3f cMax{-std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};

    for (size_t i = 0; i < N; ++i) {
        const point3u& f = faces[i];
        const point3f& v0 = vertices[f.x];
        const point3f& v1 = vertices[f.y];
        const point3f& v2 = vertices[f.z];

        // Triangle AABB
        AABBGPU aabb;
        aabb.min_x = std::min({v0.x, v1.x, v2.x});
        aabb.min_y = std::min({v0.y, v1.y, v2.y});
        aabb.min_z = std::min({v0.z, v1.z, v2.z});
        aabb.max_x = std::max({v0.x, v1.x, v2.x});
        aabb.max_y = std::max({v0.y, v1.y, v2.y});
        aabb.max_z = std::max({v0.z, v1.z, v2.z});

        // Centroid
        point3f c;
        c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
        c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
        c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

        prims[i].triIndex = static_cast<uint32_t>(i);
        prims[i].aabb     = aabb;
        prims[i].centroid = c;

        // Update centroid bounds
        cMin.x = std::min(cMin.x, c.x);
        cMin.y = std::min(cMin.y, c.y);
        cMin.z = std::min(cMin.z, c.z);
        cMax.x = std::max(cMax.x, c.x);
        cMax.y = std::max(cMax.y, c.y);
        cMax.z = std::max(cMax.z, c.z);
    }

    // 2) Compute Morton codes for centroids (normalized to [0,1]^3)
    const float eps = 1e-9f;
    const float dx = std::max(cMax.x - cMin.x, eps);
    const float dy = std::max(cMax.y - cMin.y, eps);
    const float dz = std::max(cMax.z - cMin.z, eps);

    for (size_t i = 0; i < N; ++i) {
        const point3f& c = prims[i].centroid;
        float nx = (c.x - cMin.x) / dx;
        float ny = (c.y - cMin.y) / dy;
        float nz = (c.z - cMin.z) / dz;

        // Clamp to [0,1]
        nx = std::min(std::max(nx, 0.0f), 1.0f);
        ny = std::min(std::max(ny, 0.0f), 1.0f);
        nz = std::min(std::max(nz, 0.0f), 1.0f);

        prims[i].morton = morton3D(nx, ny, nz);
    }
    */

    __shared__ float3 centroid_shared[256];

    unsigned int f1 = faces[i * 3 + 0];
    unsigned int f2 = faces[i * 3 + 1];
    unsigned int f3 = faces[i * 3 + 2];

    float3 v0 = {vertices[f1 * 3 + 0], vertices[f1 * 3 + 1], vertices[f1 * 3 + 2]};
    float3 v1 = {vertices[f2 * 3 + 0], vertices[f2 * 3 + 1], vertices[f2 * 3 + 2]};
    float3 v2 = {vertices[f3 * 3 + 0], vertices[f3 * 3 + 1], vertices[f3 * 3 + 2]};

    AABBGPU aabb_local;
    aabb_local.min_x = fminf(fminf(v0.x, v1.x), v2.x);
    aabb_local.min_y = fminf(fminf(v0.y, v1.y), v2.y);
    aabb_local.min_z = fminf(fminf(v0.z, v1.z), v2.z);
    aabb_local.max_x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
    aabb_local.max_y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
    aabb_local.max_z = fmaxf(fmaxf(v0.z, v1.z), v2.z);

    float3 centroid_local;
    centroid_local.x = (v0.x + v1.x + v2.x) / 3.0f;
    centroid_local.y = (v0.y + v1.y + v2.y) / 3.0f;
    centroid_local.z = (v0.z + v1.z + v2.z) / 3.0f;

    data_triIndex[i] = static_cast<uint32_t>(i);
    data_aabb[i]     = aabb_local;
    data_centroid[i] = centroid_local;

    centroid_shared[threadIdx.x] = centroid_local;
    __syncthreads();

    if (threadIdx.x == 0) {
        float3 cmin_local = {FLT_MAX, FLT_MAX, FLT_MAX};
        float3 cmax_local = {FLT_MIN, FLT_MIN, FLT_MIN};
        const int block_offset = blockIdx.x * blockDim.x;
        for (int i = 0; i < 256; ++i) {
            if (block_offset + i >= nFaces) {
                break;
            }

            cmin_local.x = fmin(cmin_local.x, centroid_shared[i].x);
            cmin_local.y = fmin(cmin_local.y, centroid_shared[i].y);
            cmin_local.z = fmin(cmin_local.z, centroid_shared[i].z);
            cmax_local.x = fmax(cmax_local.x, centroid_shared[i].x);
            cmax_local.y = fmax(cmax_local.y, centroid_shared[i].y);
            cmax_local.z = fmax(cmax_local.z, centroid_shared[i].z);
        }

        atomicMinFloat(&cMin->x, cmin_local.x);
        atomicMinFloat(&cMin->y, cmin_local.y);
        atomicMinFloat(&cMin->z, cmin_local.z);
        atomicMaxFloat(&cMax->x, cmax_local.x);
        atomicMaxFloat(&cMax->y, cmax_local.y);
        atomicMaxFloat(&cMax->z, cmax_local.z);
    }
}

namespace cuda {
void build_prim(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices,
    const gpu::gpu_mem_32u& faces,
    const int nVertices,
    const int nFaces,
    gpu::shared_device_buffer_typed<unsigned int>& output_data_triIndex,
    gpu::shared_device_buffer_typed<unsigned int>& output_data_morton,
    gpu::shared_device_buffer_typed<AABBGPU>& output_data_aabb,
    gpu::shared_device_buffer_typed<float3>& output_data_centroid,
    gpu::shared_device_buffer_typed<float3>& cMin,
    gpu::shared_device_buffer_typed<float3>& cMax)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::build_prim<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), 
        faces.cuptr(),
        nVertices, 
        nFaces, 
        output_data_triIndex.cuptr(),
        output_data_morton.cuptr(),
        output_data_aabb.cuptr(),
        output_data_centroid.cuptr(),
        cMin.cuptr(),
        cMax.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
