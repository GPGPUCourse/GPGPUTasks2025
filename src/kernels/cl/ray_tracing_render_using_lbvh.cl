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
    __private float*          outT, // сюда нужно записать t рассчитанный в intersect_ray_triangle(..., t, u, v)
    __private int*            outFaceId,
    __private float*          outU, // сюда нужно записать u рассчитанный в intersect_ray_triangle(..., t, u, v)
    __private float*          outV) // сюда нужно записать v рассчитанный в intersect_ray_triangle(..., t, u, v)
{
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1;

    // iterative traversal stack
    const int MAX_STACK = 64;
    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = rootIndex;

    bool hit = false;
    float bestT = FLT_MAX;
    *outFaceId = -1;
    *outT = FLT_MAX;
    *outU = 0.0f;
    *outV = 0.0f;

    while (sp > 0) {
        int nodeIndex = stack[--sp];
        BVHNodeGPU node = nodes[nodeIndex];

        float tNear, tFar;
        if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, bestT, &tNear, &tFar))
            continue;

        if (nodeIndex >= leafStart) {
            uint triIndex = leafTriIndices[nodeIndex - leafStart];

            float t, u, v;
            uint3  f = loadFace(faces, triIndex);
            float3 a = loadVertex(vertices, f.x);
            float3 b = loadVertex(vertices, f.y);
            float3 c = loadVertex(vertices, f.z);

            if (intersect_ray_triangle(orig, dir,
                                       a, b, c,
                                       tMin, bestT,
                                       false,
                                       &t, &u, &v))
            {
                hit = true;
                bestT = t;
                *outT = t;
                *outFaceId = (int)triIndex;
                *outU = u;
                *outV = v;
            }
        } else {
            int left  = (int)node.leftChildIndex;
            int right = (int)node.rightChildIndex;

            float nearL, farL;
            float nearR, farR;
            bool hitL = intersect_ray_aabb(orig, dir, nodes[left].aabb, tMin, bestT, &nearL, &farL);
            bool hitR = intersect_ray_aabb(orig, dir, nodes[right].aabb, tMin, bestT, &nearR, &farR);

            if (hitL && hitR) {
                // push farther first so nearer gets processed first
                if (nearL > nearR) {
                    stack[sp++] = left;
                    stack[sp++] = right;
                } else {
                    stack[sp++] = right;
                    stack[sp++] = left;
                }
            } else if (hitL) {
                stack[sp++] = left;
            } else if (hitR) {
                stack[sp++] = right;
            }
        }
    }

    return hit;
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

    const int MAX_STACK = 64;
    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = rootIndex;

    const float tMin = 1e-4f;
    const float tMax = FLT_MAX;

    while (sp > 0) {
        int nodeIndex = stack[--sp];
        BVHNodeGPU node = nodes[nodeIndex];

        float tNear, tFar;
        if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, tMax, &tNear, &tFar))
            continue;

        if (nodeIndex >= leafStart) {
            uint triIndex = leafTriIndices[nodeIndex - leafStart];
            if ((int)triIndex == ignore_face)
                continue;

            uint3  f = loadFace(faces, triIndex);
            float3 a = loadVertex(vertices, f.x);
            float3 b = loadVertex(vertices, f.y);
            float3 c = loadVertex(vertices, f.z);

            float t, u, v;
            if (intersect_ray_triangle(orig, dir,
                                       a, b, c,
                                       tMin, tMax,
                                       false,
                                       &t, &u, &v))
            {
                return true;
            }
        } else {
            int left  = (int)node.leftChildIndex;
            int right = (int)node.rightChildIndex;

            float nearL, farL;
            float nearR, farR;
            bool hitL = intersect_ray_aabb(orig, dir, nodes[left].aabb, tMin, tMax, &nearL, &farL);
            bool hitR = intersect_ray_aabb(orig, dir, nodes[right].aabb, tMin, tMax, &nearR, &farR);

            if (hitL && hitR) {
                if (nearL > nearR) {
                    stack[sp++] = left;
                    stack[sp++] = right;
                } else {
                    stack[sp++] = right;
                    stack[sp++] = left;
                }
            } else if (hitL) {
                stack[sp++] = left;
            } else if (hitR) {
                stack[sp++] = right;
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

/* ===================== LBVH BUILDING KERNELS ===================== */

// Morton helpers (10 bits per axis -> 30-bit code)
static inline uint expandBits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline uint morton3D(float x, float y, float z)
{
    uint ix = clamp((int)(x * 1024.0f), 0, 1023);
    uint iy = clamp((int)(y * 1024.0f), 0, 1023);
    uint iz = clamp((int)(z * 1024.0f), 0, 1023);

    uint xx = expandBits(ix);
    uint yy = expandBits(iy);
    uint zz = expandBits(iz);
    return (xx << 2) | (yy << 1) | zz;
}

// Kernel: compute Morton codes (padded length -> sentinel values)
__kernel void build_morton_codes(
    __global const float* vertices,
    __global const uint*  faces,
    uint                  nfaces,
    float                 min_x,
    float                 min_y,
    float                 min_z,
    float                 max_x,
    float                 max_y,
    float                 max_z,
    uint                  padded_length,
    __global uint*        morton_codes,
    __global uint*        tri_indices)
{
    uint idx = get_global_id(0);
    if (idx >= padded_length)
        return;

    if (idx >= nfaces) {
        morton_codes[idx] = 0xFFFFFFFFu;
        tri_indices[idx]  = 0u;
        return;
    }

    uint3 f = loadFace(faces, idx);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    float3 c = (float3)((v0.x + v1.x + v2.x) * (1.0f / 3.0f),
                        (v0.y + v1.y + v2.y) * (1.0f / 3.0f),
                        (v0.z + v1.z + v2.z) * (1.0f / 3.0f));

    float dx = fmax(max_x - min_x, 1e-9f);
    float dy = fmax(max_y - min_y, 1e-9f);
    float dz = fmax(max_z - min_z, 1e-9f);

    float nx = clamp((c.x - min_x) / dx, 0.0f, 1.0f);
    float ny = clamp((c.y - min_y) / dy, 0.0f, 1.0f);
    float nz = clamp((c.z - min_z) / dz, 0.0f, 1.0f);

    morton_codes[idx] = morton3D(nx, ny, nz);
    tri_indices[idx]  = idx;
}

// Bitonic sort step for (key, value)
__kernel void bitonic_sort_step(
    __global uint* keys,
    __global uint* values,
    uint           j,
    uint           k,
    uint           length)
{
    uint i = get_global_id(0);
    if (i >= length)
        return;

    uint ixj = i ^ j;
    if (ixj <= i || ixj >= length)
        return;

    uint key_i = keys[i];
    uint key_j = keys[ixj];
    uint val_i = values[i];
    uint val_j = values[ixj];

    bool ascending = ((i & k) == 0);
    bool swap = ascending ? (key_i > key_j) : (key_i < key_j);
    if (key_i == key_j) {
        swap = ascending ? (val_i > val_j) : (val_i < val_j);
    }

    if (swap) {
        keys[i]   = key_j;
        keys[ixj] = key_i;
        values[i] = val_j;
        values[ixj] = val_i;
    }
}

// Copy sorted triangle indices into leafTriIndices buffer
__kernel void copy_leaf_indices(__global const uint* sorted_tri_indices,
                                __global       uint*  leaf_tri_indices,
                                uint                  nfaces)
{
    uint idx = get_global_id(0);
    if (idx >= nfaces) return;
    leaf_tri_indices[idx] = sorted_tri_indices[idx];
}

// Build leaf nodes from sorted triangles
__kernel void build_leaf_nodes(
    __global const float* vertices,
    __global const uint*  faces,
    __global const uint*  sorted_tri_indices,
    __global       BVHNodeGPU* nodes,
    uint nfaces)
{
    uint idx = get_global_id(0);
    if (idx >= nfaces) return;

    uint tri = sorted_tri_indices[idx];
    uint3 f = loadFace(faces, tri);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    AABBGPU aabb;
    aabb.min_x = fmin(fmin(v0.x, v1.x), v2.x);
    aabb.min_y = fmin(fmin(v0.y, v1.y), v2.y);
    aabb.min_z = fmin(fmin(v0.z, v1.z), v2.z);
    aabb.max_x = fmax(fmax(v0.x, v1.x), v2.x);
    aabb.max_y = fmax(fmax(v0.y, v1.y), v2.y);
    aabb.max_z = fmax(fmax(v0.z, v1.z), v2.z);

    uint leafIndex = (nfaces - 1u) + idx;
    nodes[leafIndex].aabb = aabb;
    nodes[leafIndex].leftChildIndex  = 0xFFFFFFFFu;
    nodes[leafIndex].rightChildIndex = 0xFFFFFFFFu;
}

// Prefix helpers
static inline int clz32(uint x) { return (int)clz(x); }

static inline int common_prefix(__global const uint* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;
    uint ci = codes[i];
    uint cj = codes[j];
    if (ci == cj) {
        uint diff = ((uint)i) ^ ((uint)j);
        return 32 + clz32(diff);
    } else {
        uint diff = ci ^ cj;
        return clz32(diff);
    }
}

static inline void determine_range(__global const uint* codes, int N, int i, __private int* outFirst, __private int* outLast)
{
    int cpL = common_prefix(codes, N, i, i - 1);
    int cpR = common_prefix(codes, N, i, i + 1);
    int d = (cpR > cpL) ? 1 : -1;

    int deltaMin = common_prefix(codes, N, i, i - d);
    int lmax = 2;
    while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }

    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * d;
    *outFirst = min(i, j);
    *outLast  = max(i, j);
}

static inline int find_split(__global const uint* codes, int first, int last)
{
    if (first == last) return first;
    int commonPrefix = common_prefix(codes, (last + 1), first, last);
    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;
        if (newSplit < last) {
            int splitPrefix = common_prefix(codes, (last + 1), first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

// Build internal nodes (child indices)
__kernel void build_internal_nodes(__global const uint* morton_codes,
                                   __global BVHNodeGPU* nodes,
                                   uint                 nfaces)
{
    uint idx = get_global_id(0);
    if (idx >= nfaces - 1u)
        return;

    int first, last;
    determine_range(morton_codes, (int)nfaces, (int)idx, &first, &last);
    int split = find_split(morton_codes, first, last);

    int leftIndex = (split == first) ? ((int)(nfaces - 1) + split) : split;
    int rightIndex = (split + 1 == last) ? ((int)(nfaces - 1) + split + 1) : (split + 1);

    nodes[idx].leftChildIndex  = (uint)leftIndex;
    nodes[idx].rightChildIndex = (uint)rightIndex;
}

// Init internal nodes AABB with sentinel values
__kernel void init_internal_aabb(__global BVHNodeGPU* nodes, uint nfaces)
{
    uint idx = get_global_id(0);
    if (idx >= nfaces - 1u)
        return;

    nodes[idx].aabb.min_x = FLT_MAX;
    nodes[idx].aabb.min_y = FLT_MAX;
    nodes[idx].aabb.min_z = FLT_MAX;
    nodes[idx].aabb.max_x = -FLT_MAX;
    nodes[idx].aabb.max_y = -FLT_MAX;
    nodes[idx].aabb.max_z = -FLT_MAX;
}

// Propagate AABB upwards (iterate several times from host)
__kernel void propagate_aabb(__global BVHNodeGPU* nodes, uint nfaces)
{
    uint idx = get_global_id(0);
    if (idx >= nfaces - 1u)
        return;

    BVHNodeGPU node = nodes[idx];
    BVHNodeGPU left  = nodes[node.leftChildIndex];
    BVHNodeGPU right = nodes[node.rightChildIndex];

    AABBGPU aabb;
    aabb.min_x = fmin(left.aabb.min_x, right.aabb.min_x);
    aabb.min_y = fmin(left.aabb.min_y, right.aabb.min_y);
    aabb.min_z = fmin(left.aabb.min_z, right.aabb.min_z);
    aabb.max_x = fmax(left.aabb.max_x, right.aabb.max_x);
    aabb.max_y = fmax(left.aabb.max_y, right.aabb.max_y);
    aabb.max_z = fmax(left.aabb.max_z, right.aabb.max_z);

    nodes[idx].aabb = aabb;
}
