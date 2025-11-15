#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../defines.h"
#include "helpers/rassert.cu"

#include "camera_helpers.cu"
#include "geometry_helpers.cu"
#include "random_helpers.cu"

// BVH traversal: closest hit along ray
__device__ bool bvh_closest_hit(
    const float3& orig,
    const float3& dir,
    const BVHNodeGPU* nodes,
    const unsigned int* leafTriIndices,
    unsigned int nfaces,
    const float* vertices,
    const unsigned int* faces,
    float tMin,
    float& outT,
    int& outFaceId,
    float& outU,
    float& outV)
{
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1; // leaves: [leafStart .. leafStart + nfaces - 1]

    // Simple fixed-size stack (sufficient for typical BVHs)
    int stack[64];
    int stackSize = 0;
    stack[stackSize++] = rootIndex;

    float tBest = FLT_MAX;
    int   faceBest = -1;
    float uBest = 0.0f, vBest = 0.0f;

    float tNear, tFar;

    while (stackSize > 0) {
        int nodeIndex = stack[--stackSize];
        const BVHNodeGPU& node = nodes[nodeIndex];

        // AABB culling with current best t
        if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, tBest, tNear, tFar))
            continue;

        bool isLeaf = (nodeIndex >= leafStart);
        if (isLeaf) {
            unsigned int leafIdx  = (unsigned int)(nodeIndex - leafStart);
            unsigned int triIndex = leafTriIndices[leafIdx];

            uint3 f = loadFace(faces, triIndex);
            float3 v0 = loadVertex(vertices, f.x);
            float3 v1 = loadVertex(vertices, f.y);
            float3 v2 = loadVertex(vertices, f.z);

            float t,u,v;
            // backface culling: false (same as original kernel)
            if (intersect_ray_triangle(orig, dir, v0, v1, v2,
                                       tMin, tBest, false, t,u,v))
            {
                if (t < tBest) {
                    tBest = t;
                    faceBest = (int)triIndex;
                    uBest = u;
                    vBest = v;
                }
            }
        } else {
            // Internal node: push children
            // (no near/far sorting for simplicity)
            if (stackSize < 62) { // avoid overflow
                stack[stackSize++] = (int)node.leftChildIndex;
                stack[stackSize++] = (int)node.rightChildIndex;
            }
        }
    }

    if (faceBest >= 0) {
        outT       = tBest;
        outFaceId  = faceBest;
        outU       = uBest;
        outV       = vBest;
        return true;
    }
    return false;
}

// BVH traversal: any hit (for AO rays)
// ignore_face: do not count intersections with this triangle index
__device__ bool any_hit_from(
    const float3& orig,
    const float3& dir,
    const float* vertices,
    const unsigned int* faces,
    const BVHNodeGPU* nodes,
    const unsigned int* leafTriIndices,
    unsigned int nfaces,
    int ignore_face)
{
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1;

    int stack[64];
    int stackSize = 0;
    stack[stackSize++] = rootIndex;

    const float tMin = 1e-4f;
    const float tMax = FLT_MAX;
    float tNear, tFar;

    while (stackSize > 0) {
        int nodeIndex = stack[--stackSize];
        const BVHNodeGPU& node = nodes[nodeIndex];

        if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, tMax, tNear, tFar))
            continue;

        bool isLeaf = (nodeIndex >= leafStart);
        if (isLeaf) {
            unsigned int leafIdx  = (unsigned int)(nodeIndex - leafStart);
            unsigned int triIndex = leafTriIndices[leafIdx];
            if ((int)triIndex == ignore_face)
                continue;

            uint3 f = loadFace(faces, triIndex);
            float3 a = loadVertex(vertices, f.x);
            float3 b = loadVertex(vertices, f.y);
            float3 c = loadVertex(vertices, f.z);

            float t,u,v;
            if (intersect_ray_triangle(orig, dir, a, b, c,
                                       tMin, tMax, false, t,u,v))
            {
                return true; // early out on first occluder
            }
        } else {
            if (stackSize < 62) {
                stack[stackSize++] = (int)node.leftChildIndex;
                stack[stackSize++] = (int)node.rightChildIndex;
            }
        }
    }
    return false;
}

// + helper: build tangent basis for a given normal
__device__ inline void make_basis(const float3& n, float3& t, float3& b) {
    // pick a non-parallel vector
    float3 up = (fabsf(n.z) < 0.999f) ? make_float3(0.f,0.f,1.f) : make_float3(0.f,1.f,0.f);
    t = normalize_f3(cross_f3(up, n));
    b = cross_f3(n, t);
}

__global__ void ray_tracing_render_using_lbvh(
    const float* vertices,
    const unsigned int* faces,
    const BVHNodeGPU* bvhNodes,
    const unsigned int* leafTriIndices,
    int* framebuffer_face_id,
    float* framebuffer_ambient_occlusion,
    CameraViewGPU *camera,
    unsigned int nfaces
)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    curassert(camera.magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 946435342);
    if (i >= camera->K.width || j >= camera->K.height)
        return;

    float3 ray_origin, ray_direction;
    make_primary_ray(*camera, i + 0.5f, j + 0.5f, ray_origin, ray_direction);

    float tMin  = 1e-6f;
    float tBest = FLT_MAX;
    float uBest=0, vBest=0;
    int   faceIdBest = -1;

    // Use BVH traversal instead of brute-force loop
    bool hit = bvh_closest_hit(
        ray_origin,
        ray_direction,
        bvhNodes,
        leafTriIndices,
        nfaces,
        vertices,
        faces,
        tMin,
        tBest,
        faceIdBest,
        uBest,
        vBest
    );

    const unsigned int idx = j * camera->K.width + i;
    framebuffer_face_id[idx] = faceIdBest;

    float ao = 1.0f; // background stays white
    if (faceIdBest >= 0) {
        uint3 f = loadFace(faces, faceIdBest);
        float3 a = loadVertex(vertices, f.x);
        float3 b = loadVertex(vertices, f.y);
        float3 c = loadVertex(vertices, f.z);

        float3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
        float3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
        float3 n  = normalize_f3(cross_f3(e1, e2));

        // ensure hemisphere is "outside" relative to the camera ray
        if (n.x*ray_direction.x + n.y*ray_direction.y + n.z*ray_direction.z > 0.0f)
            n = make_float3(-n.x, -n.y, -n.z);

        float3 P  = {ray_origin.x + tBest*ray_direction.x,
            ray_origin.y + tBest*ray_direction.y,
            ray_origin.z + tBest*ray_direction.z};

        float scale = fmaxf(fmaxf(length_f3(e1), length_f3(e2)),
            length_f3(make_float3(c.x-a.x, c.y-a.y, c.z-a.z)));
        float eps   = 1e-3f * fmaxf(1.0f, scale);
        float3 Po   = {P.x + n.x*eps, P.y + n.y*eps, P.z + n.z*eps};

        // build tangent basis
        float3 T, B;
        make_basis(n, T, B);
        // per-pixel seed (stable)
        union {
            float f32;
            uint32_t u32;
        } tBestUnion;
        tBestUnion.f32 = tBest;
        uint32_t rng = 0x9E3779B9u ^ idx ^ tBestUnion.u32;

        int hits = 0;
        for (int s = 0; s < AO_SAMPLES; ++s) {
            // uniform hemisphere sampling (solid angle)
            float u1 = random01(rng);
            float u2 = random01(rng);
            float z  = u1;                          // z in [0,1]
            float phi = 6.28318530718f * u2;        // 2*pi*u2
            float r  = sqrtf(fmaxf(0.f, 1.f - z*z));
            float3 d_local = make_float3(r * cosf(phi), r * sinf(phi), z);

            // transform to world space
            float3 d = make_float3(
                T.x*d_local.x + B.x*d_local.y + n.x*d_local.z,
                T.y*d_local.x + B.y*d_local.y + n.y*d_local.z,
                T.z*d_local.x + B.z*d_local.y + n.z*d_local.z
            );

            if (any_hit_from(Po, d,
                             vertices, faces,
                             bvhNodes, leafTriIndices,
                             nfaces, faceIdBest))
                ++hits;
        }
        ao = 1.0f - (float)hits / (float)AO_SAMPLES; // [0,1]
    }
    framebuffer_ambient_occlusion[idx] = ao;
}


namespace cuda {
void ray_tracing_render_using_lbvh(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU> &bvhNodes, const gpu::gpu_mem_32u &leafTriIndices,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::ray_tracing_render_using_lbvh<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        vertices.cuptr(), faces.cuptr(),
        bvhNodes.cuptr(), leafTriIndices.cuptr(),
        framebuffer_face_id.cuptr(), framebuffer_ambient_occlusion.cuptr(),
        camera.cuptr(),
        nfaces
        );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
