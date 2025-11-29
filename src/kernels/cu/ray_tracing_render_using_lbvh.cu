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

#define STACK_SIZE 128

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
    float& outT, // сюда нужно записать t рассчитанный в intersect_ray_triangle(..., t, u, v)
    int& outFaceId,
    float& outU, // сюда нужно записать u рассчитанный в intersect_ray_triangle(..., t, u, v)
    float& outV) // сюда нужно записать v рассчитанный в intersect_ray_triangle(..., t, u, v)
{
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1;

    int stack[STACK_SIZE];
    int stackSize = 1;
    stack[0] = rootIndex;

    float tMinStack[STACK_SIZE];
    float tMaxStack[STACK_SIZE];

    tMinStack[0] = tMin;
    tMaxStack[0] = FLT_MAX;

    float bestT = FLT_MAX;
    float t, u, v;

    while (stackSize > 0) {
        stackSize--;
        int nodeIndex = stack[stackSize];
        BVHNodeGPU node = nodes[nodeIndex];

        float currentTMin = tMinStack[stackSize];
        float currentTMax = tMaxStack[stackSize];

        if (nodeIndex >= leafStart) {
            int triIndex = leafTriIndices[nodeIndex - leafStart];
            uint3 face = loadFace(faces, triIndex);
            float3 a = loadVertex(vertices, face.x);
            float3 b = loadVertex(vertices, face.y);
            float3 c = loadVertex(vertices, face.z);
            
            bool triangleHit = intersect_ray_triangle(orig, dir, a, b, c, currentTMin, currentTMax, false, t, u, v);

            if (triangleHit) {
                if (t < bestT) {
                    bestT = t;
                    outFaceId = triIndex;
                    outU = u;
                    outV = v;
                    outT = t;
                }
            }
        } else {
            float tHitNear, tHitFar;
            bool hit = intersect_ray_aabb(orig, dir, node.aabb, currentTMin, currentTMax, tHitNear, tHitFar);

            if (hit) {
                currentTMin = tHitNear;
                currentTMax = tHitFar;

                tMinStack[stackSize] = 0.0;
                tMaxStack[stackSize] = FLT_MAX;
                tMinStack[stackSize + 1] = 0.0;
                tMaxStack[stackSize + 1] = FLT_MAX;

                stack[stackSize] = node.leftChildIndex;
                stack[stackSize + 1] = node.rightChildIndex;
                stackSize += 2;
            }
        }
    }

    return bestT < FLT_MAX;
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

    int stack[STACK_SIZE];
    int stackSize = 1;
    stack[0] = rootIndex;

    float tMinStack[STACK_SIZE];
    float tMaxStack[STACK_SIZE];

    tMinStack[0] = 0.0;
    tMaxStack[0] = FLT_MAX;

    while (stackSize > 0) {
        stackSize--;
        int nodeIndex = stack[stackSize];
        BVHNodeGPU node = nodes[nodeIndex];

        float currentTMin = tMinStack[stackSize];
        float currentTMax = tMaxStack[stackSize];

        if (nodeIndex >= leafStart) {
            int triIndex = leafTriIndices[nodeIndex - leafStart];

            if (triIndex == ignore_face) {
                continue;
            }

            uint3 face = loadFace(faces, triIndex);
            float3 a = loadVertex(vertices, face.x);
            float3 b = loadVertex(vertices, face.y);
            float3 c = loadVertex(vertices, face.z);

            float t, u, v;
            bool triangleHit = intersect_ray_triangle(orig, dir, a, b, c, currentTMin, currentTMax, false, t, u, v);

            if (triangleHit) {
                return true;
            }
        } else {
            float tHitNear, tHitFar;
            bool hit = intersect_ray_aabb(orig, dir, node.aabb, currentTMin, currentTMax, tHitNear, tHitFar);

            if (hit) {
                currentTMin = tHitNear;
                currentTMax = tHitFar;

                tMinStack[stackSize] = 0.0;
                tMaxStack[stackSize] = FLT_MAX;
                tMinStack[stackSize + 1] = 0.0;
                tMaxStack[stackSize + 1] = FLT_MAX;

                stack[stackSize] = node.leftChildIndex;
                stack[stackSize + 1] = node.rightChildIndex;
                stackSize += 2;
            }
        }
    }

    return false; // no intersections found
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
