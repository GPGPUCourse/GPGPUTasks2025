#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

// BVH traversal: closest hit along ray
static inline bool bvh_closest_hit(
    const float3              orig,
    const float3              dir,
    __global const BVHNodeGPU* nodes,
    __global const uint*      leafTriIndices,
    uint                      nfaces,
    __global const float*     vertices,
    __global const uint*      faces,
    float                     tMin,
    __private float*          outT,
    __private int*            outFaceId,
    __private float*          outU,
    __private float*          outV)
{
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1; // leaves: [leafStart .. leafStart + nfaces - 1]

    // Simple fixed-size stack (sufficient for typical BVHs)
    int stack[64];
    int stackSize = 0;
    stack[stackSize++] = rootIndex;

    float tBest   = FLT_MAX;
    int   faceBest = -1;
    float uBest   = 0.0f;
    float vBest   = 0.0f;

    float tNear;
    float tFar;

    while (stackSize > 0) {
        int nodeIndex = stack[--stackSize];
        BVHNodeGPU node = nodes[nodeIndex];

        // AABB culling with current best t
        if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, tBest, &tNear, &tFar))
            continue;

        bool isLeaf = (nodeIndex >= leafStart);
        if (isLeaf) {
            uint leafIdx  = (uint)(nodeIndex - leafStart);
            uint triIndex = leafTriIndices[leafIdx];

            uint3  f  = loadFace(faces, triIndex);
            float3 v0 = loadVertex(vertices, f.x);
            float3 v1 = loadVertex(vertices, f.y);
            float3 v2 = loadVertex(vertices, f.z);

            float t;
            float u;
            float v;
            // backface culling: false (same as original kernel)
            if (intersect_ray_triangle(orig, dir, v0, v1, v2,
                                       tMin, tBest, false,
                                       &t, &u, &v))
            {
                if (t < tBest) {
                    tBest    = t;
                    faceBest = (int)triIndex;
                    uBest    = u;
                    vBest    = v;
                }
            }
        } else {
            // Internal node: push children (no near/far sorting for simplicity)
            if (stackSize < 62) { // avoid overflow
                stack[stackSize++] = (int)node.leftChildIndex;
                stack[stackSize++] = (int)node.rightChildIndex;
            }
        }
    }

    if (faceBest >= 0) {
        *outT      = tBest;
        *outFaceId = faceBest;
        *outU      = uBest;
        *outV      = vBest;
        return true;
    }

    return false;
}

// Cast a single ray and report if ANY occluder is hit (for ambient occlusion)
static inline bool any_hit_from(
    const float3              orig,
    const float3              dir,
    __global const float*     vertices,
    __global const uint*      faces,
    __global const BVHNodeGPU* nodes,
    __global const uint*      leafTriIndices,
    uint                      nfaces,
    int                       ignore_face)
{
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1;

    int stack[64];
    int stackSize = 0;
    stack[stackSize++] = rootIndex;

    const float tMin = 1e-4f;
    const float tMax = FLT_MAX;
    float tNear;
    float tFar;

    while (stackSize > 0) {
        int nodeIndex = stack[--stackSize];
        BVHNodeGPU node = nodes[nodeIndex];

        if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, tMax, &tNear, &tFar))
            continue;

        bool isLeaf = (nodeIndex >= leafStart);
        if (isLeaf) {
            uint leafIdx  = (uint)(nodeIndex - leafStart);
            uint triIndex = leafTriIndices[leafIdx];
            if ((int)triIndex == ignore_face)
                continue;

            uint3  f = loadFace(faces, triIndex);
            float3 a = loadVertex(vertices, f.x);
            float3 b = loadVertex(vertices, f.y);
            float3 c = loadVertex(vertices, f.z);

            float t;
            float u;
            float v;
            if (intersect_ray_triangle(orig, dir, a, b, c,
                                       tMin, tMax, false,
                                       &t, &u, &v))
            {
                // early out on first occluder
                return true;
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

// Helper: build tangent basis for a given normal
static inline void make_basis(const float3 n,
                              __private float3* t,
                              __private float3* b)
{
    // pick a non-parallel vector
    float3 up = (fabs(n.z) < 0.999f)
        ? (float3)(0.0f, 0.0f, 1.0f)
        : (float3)(0.0f, 1.0f, 0.0f);

    *t = normalize_f3(cross_f3(up, n));
    *b = cross_f3(n, *t);
}

__kernel void ray_tracing_render_using_lbvh(
    __global const float*      vertices,
    __global const uint*       faces,
    __global const BVHNodeGPU* bvhNodes,
    __global const uint*       leafTriIndices,
    __global int*              framebuffer_face_id,
    __global float*            framebuffer_ambient_occlusion,
    __global const CameraViewGPU* camera,
    uint                       nfaces)
{
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    rassert(camera.magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 786435342);
    if (i >= camera->K.width || j >= camera->K.height)
        return;

    float3 ray_origin;
    float3 ray_direction;
    make_primary_ray(camera,
                     (float)i + 0.5f,
                     (float)j + 0.5f,
                     &ray_origin,
                     &ray_direction);

    float tMin      = 1e-6f;
    float tBest     = FLT_MAX;
    float uBest     = 0.0f;
    float vBest     = 0.0f;
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
        &tBest,
        &faceIdBest,
        &uBest,
        &vBest);

    const uint idx = j * camera->K.width + i;
    framebuffer_face_id[idx] = faceIdBest;

    float ao = 1.0f; // background stays white

    if (faceIdBest >= 0) {
        uint3  f = loadFace(faces, (uint)faceIdBest);
        float3 a = loadVertex(vertices, f.x);
        float3 b = loadVertex(vertices, f.y);
        float3 c = loadVertex(vertices, f.z);

        float3 e1 = (float3)(b.x - a.x, b.y - a.y, b.z - a.z);
        float3 e2 = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float3 n  = normalize_f3(cross_f3(e1, e2));

        // ensure hemisphere is "outside" relative to the camera ray
        if (n.x * ray_direction.x +
            n.y * ray_direction.y +
            n.z * ray_direction.z > 0.0f)
        {
            n = (float3)(-n.x, -n.y, -n.z);
        }

        float3 P = (float3)(ray_origin.x + tBest * ray_direction.x,
                            ray_origin.y + tBest * ray_direction.y,
                            ray_origin.z + tBest * ray_direction.z);

        float3 ac = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float  scale = fmax(fmax(length_f3(e1), length_f3(e2)),
                            length_f3(ac));

        float  eps = 1e-3f * fmax(1.0f, scale);
        float3 Po  = (float3)(P.x + n.x * eps,
                              P.y + n.y * eps,
                              P.z + n.z * eps);

        // build tangent basis
        float3 T;
        float3 B;
        make_basis(n, &T, &B);

        // per-pixel seed (stable)
        union {
            float f32;
            uint  u32;
        } tBestUnion;
        tBestUnion.f32 = tBest;
        uint rng = (uint)(0x9E3779B9u ^ idx ^ tBestUnion.u32);

        int hits = 0;
        for (int s = 0; s < AO_SAMPLES; ++s) {
            // uniform hemisphere sampling (solid angle)
            float u1  = random01(&rng);
            float u2  = random01(&rng);
            float z   = u1;                      // z in [0,1]
            float phi = 6.28318530718f * u2;     // 2*pi*u2
            float r   = sqrt(fmax(0.0f, 1.0f - z * z));
            float3 d_local = (float3)(r * cos(phi),
                                      r * sin(phi),
                                      z);

            // transform to world space
            float3 d = (float3)(
                T.x * d_local.x + B.x * d_local.y + n.x * d_local.z,
                T.y * d_local.x + B.y * d_local.y + n.y * d_local.z,
                T.z * d_local.x + B.z * d_local.y + n.z * d_local.z
            );

            if (any_hit_from(Po, d,
                             vertices, faces,
                             bvhNodes, leafTriIndices,
                             nfaces, faceIdBest))
            {
                ++hits;
            }
        }

        ao = 1.0f - (float)hits / (float)AO_SAMPLES; // [0,1]
    }

    framebuffer_ambient_occlusion[idx] = ao;
}
