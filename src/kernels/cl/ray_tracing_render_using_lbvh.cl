#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

#define MAX_STACK_SIZE 100

// BVH traversal: closest hit along ray
static inline bool bvh_closest_hit(
    const float3              orig,
    const float3              dir,
    __global const BVHNodeGPU* nodes, // узлы BVH - 0 = корень
    __global const uint*      leafTriIndices, // index треугольника в листе BVH
    uint                      nfaces, // кол-во треугольников
    __global const float*     vertices, // вершины - Vn.x, Vn.y, Vn.z, V(n+1).x, ...
    __global const uint*      faces, // треугольники - Tn.VIndex1, Tn.VIndex2, Tn.VIndex3, T(n+1).VIndex1, ...
    float                     tMin,
    __private float*          outT, // сюда нужно записать t рассчитанный в intersect_ray_triangle(..., t, u, v)
    __private int*            outFaceId,
    __private float*          outU, // сюда нужно записать u рассчитанный в intersect_ray_triangle(..., t, u, v)
    __private float*          outV) // сюда нужно записать v рассчитанный в intersect_ray_triangle(..., t, u, v)
{
    const uint leafStart = nfaces - 1;

    uint stack[MAX_STACK_SIZE];
    uint stack_size = 0;
    stack[stack_size++] = 0; // добавим root в стек

    do {
        const uint node_index = stack[--stack_size]; // достаём узел со стека

        if (node_index >= leafStart) { // текущий узел - лист
            const uint triIndex = leafTriIndices[node_index - leafStart];

            const uint3  face = loadFace(faces, triIndex);
            const float3 v0 = loadVertex(vertices, face.x);
            const float3 v1 = loadVertex(vertices, face.y);
            const float3 v2 = loadVertex(vertices, face.z);

            float t, u, v;
            if (intersect_ray_triangle(
                orig, dir,
                v0, v1, v2,
                tMin, *outT, false,
                &t, &u, &v))
            {
                *outT = t;
                *outFaceId = triIndex;
                *outU = u;
                *outV = v;
            }
        } else {
            __global const BVHNodeGPU* node = &nodes[node_index];

            float tHitNear[3], tHitFar[3]; // unused
            if (intersect_ray_aabb(
                orig, dir,
                node->aabb,
                tMin, FLT_MAX,
                tHitNear, tHitFar))
            {
                if (node->leftChildIndex != UINT_MAX) {
                    stack[stack_size++] = node->leftChildIndex;
                }
                if (node->rightChildIndex != UINT_MAX) {
                    stack[stack_size++] = node->rightChildIndex;
                }
            }
        }

    } while (stack_size > 0);

    return *outFaceId != -1;
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
    const uint leafStart = nfaces - 1;

    uint stack[MAX_STACK_SIZE];
    uint stack_size = 0;
    stack[stack_size++] = 0; // добавим root в стек

    do {
        const uint node_index = stack[--stack_size]; // достаём узел со стека

        if (node_index >= leafStart) { // текущий узел - лист
            const uint triIndex = leafTriIndices[node_index - leafStart];

            const uint3  face = loadFace(faces, triIndex);
            const float3 v0 = loadVertex(vertices, face.x);
            const float3 v1 = loadVertex(vertices, face.y);
            const float3 v2 = loadVertex(vertices, face.z);

            float t, u, v;
            if (intersect_ray_triangle(
                orig, dir,
                v0, v1, v2,
                1e-4f, FLT_MAX, false,
                &t, &u, &v))
            {
                return true;
            }
        } else {
            __global const BVHNodeGPU* node = &nodes[node_index];

            float tHitNear[3], tHitFar[3]; // unused
            if (intersect_ray_aabb(
                orig, dir,
                node->aabb,
                1e-4f, FLT_MAX,
                tHitNear, tHitFar))
            {
                if (node->leftChildIndex != UINT_MAX) {
                    stack[stack_size++] = node->leftChildIndex;
                }
                if (node->rightChildIndex != UINT_MAX) {
                    stack[stack_size++] = node->rightChildIndex;
                }
            }
        }

    } while (stack_size > 0);

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

static inline float max(
    const float x,
    const float y,
    const float z)
{
    return max(x, max(y, z));
}

static inline float min(
    const float x,
    const float y,
    const float z)
{
    return min(x, min(y, z));
}

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
static inline uint expandBits(uint v)
{
    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
static inline MortonCode morton3D(
    const float x,
    const float y,
    const float z)
{
    // Map and clamp to integer grid [0, 1023]
    const uint ix = min(max((int)(x * 1024.0f), 0), 1023);
    const uint iy = min(max((int)(y * 1024.0f), 0), 1023);
    const uint iz = min(max((int)(z * 1024.0f), 0), 1023);

    const uint xx = expandBits(ix);
    const uint yy = expandBits(iy);
    const uint zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void pre_build_lbvh(
    __global const float      * vertices,
    __global const uint       * faces,
    __global       Prim       * prims, // size = N
             const uint         N)     //      > 2
{
    const uint index = get_global_id(0);

    if (index >= N) {
        return;
    }

    // 1) Compute per-triangle AABB and centroids
    float3 cMin;
    cMin.x = +INFINITY;
    cMin.y = +INFINITY;
    cMin.z = +INFINITY;

    float3 cMax;
    cMax.x = -INFINITY;
    cMax.y = -INFINITY;
    cMax.z = -INFINITY;

    __global Prim * prim = prims + index;

    // Centroid
    float3 c;

    {
        const uint3 face = loadFace(faces, index);
        const float3 v0 = loadVertex(vertices, face.x);
        const float3 v1 = loadVertex(vertices, face.y);
        const float3 v2 = loadVertex(vertices, face.z);

        // Triangle AABB
        AABBGPU aabb;
        aabb.min_x = min(v0.x, v1.x, v2.x);
        aabb.min_y = min(v0.y, v1.y, v2.y);
        aabb.min_z = min(v0.z, v1.z, v2.z);
        aabb.max_x = max(v0.x, v1.x, v2.x);
        aabb.max_y = max(v0.y, v1.y, v2.y);
        aabb.max_z = max(v0.z, v1.z, v2.z);

        c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
        c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
        c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

        prim->triIndex = index;
        prim->aabb     = aabb;

        // Update centroid bounds
        cMin.x = min(cMin.x, c.x);
        cMin.y = min(cMin.y, c.y);
        cMin.z = min(cMin.z, c.z);
        cMax.x = max(cMax.x, c.x);
        cMax.y = max(cMax.y, c.y);
        cMax.z = max(cMax.z, c.z);
    }

    // 2) Compute Morton codes for centroids (normalized to [0,1]^3)
    const float eps = 1e-9f;
    const float dx = max(cMax.x - cMin.x, eps);
    const float dy = max(cMax.y - cMin.y, eps);
    const float dz = max(cMax.z - cMin.z, eps);

    {
        float nx = (c.x - cMin.x) / dx;
        float ny = (c.y - cMin.y) / dy;
        float nz = (c.z - cMin.z) / dz;

        // Clamp to [0,1]
        nx = min(max(nx, 0.0f), 1.0f);
        ny = min(max(ny, 0.0f), 1.0f);
        nz = min(max(nz, 0.0f), 1.0f);

        prim->morton = morton3D(nx, ny, nz);
    }
}

static inline int common_prefix(
    __global const Prim * codes, 
             const int    N,
             const int    i,
             const int    j)
{
    if (j < 0 || j >= N) {
        return -1;
    }

    const MortonCode ci = codes[i].morton;
    const MortonCode cj = codes[j].morton;

    if (ci == cj) {
        const uint di = i;
        const uint dj = j;
        const uint diff = di ^ dj;
        return 32 + clz(diff);
    } else {
        const uint diff = ci ^ cj;
        return clz(diff);
    }
}

// Determine range [first, last] of primitives covered by internal node i
static inline void determine_range(
    __global const Prim * codes,
             const int    N,
             const int    i,
                   int  * outFirst,
                   int  * outLast)
{
    const int cpL = common_prefix(codes, N, i, i - 1);
    const int cpR = common_prefix(codes, N, i, i + 1);

    // Direction of the range
    const int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    const int deltaMin = common_prefix(codes, N, i, i - d);
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

    const int j = i + l * d;
    *outFirst = min(i, j);
    *outLast  = max(i, j);
}

// Find split position inside range [first, last] using the same
// prefix metric as determine_range (code + index tie-break)
static inline int find_split(
    __global const Prim * codes,
             const int first,
             const int last,
             const int N)
{
    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last) {
        return first;
    }

    // Prefix between first and last (уже с учётом индекса, если коды равны)
    const int commonPrefix = common_prefix(codes, N, first, last);

    int split = first;
    int step  = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
    do {
        step = (step + 1) >> 1;
        const int newSplit = split + step;

        if (newSplit < last) {
            const int splitPrefix = common_prefix(codes, N, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

// Build LBVH (Karras 2013) on CPU.
// Output:
//   outNodes           - BVH nodes array of size (2*N - 1). Root is node 0.
//   outLeafTriIndices  - size N, mapping leaf i -> original triangle index.
//
// Node indexing convention (matches LBVH style):
//   N = number of triangles (faces.size()).
//   Internal nodes: indices [0 .. N-2]
//   Leaf nodes:     indices [N-1 .. 2*N-2]
//   Leaf at index (N-1 + i) corresponds to outLeafTriIndices[i].
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void build_lbvh(
    __global const float      * vertices,
    __global const uint       * faces,
    __global       BVHNodeGPU * outNodes,
    __global       uint       * outLeafTriIndices,
    __global const Prim       * prims,
             const int          N)
{
    const uint index = get_global_id(0);
    
    if (index >= N) {
        return;
    }

    // 4) Prepare array
    {
        outLeafTriIndices[index] = prims[index].triIndex;
    }

    // 5) Initialize leaf nodes [N-1 .. 2*N-2]
    const GPUC_UINT INVALID = UINT_MAX;

    {
        uint leafIndex = (N - 1) + index;
        __global BVHNodeGPU * leaf = &outNodes[leafIndex];

        leaf->aabb = prims[index].aabb;
        leaf->leftChildIndex  = INVALID;
        leaf->rightChildIndex = INVALID;
    }

    // 6) Build internal nodes [0 .. N-2]
    if (index < N - 1) {
        int first, last;
        determine_range(prims, N, index, &first, &last);
        int split = find_split(prims, first, last, N);

        // Left child
        int leftIndex;
        if (split == first) {
            // Range [first, split] has one primitive -> leaf
            leftIndex = (N - 1) + split;
        } else {
            // Internal node
            leftIndex = split;
        }

        // Right child
        int rightIndex;
        if (split + 1 == last) {
            // Range [split+1, last] has one primitive -> leaf
            rightIndex = (N - 1) + split + 1;
        } else {
            // Internal node
            rightIndex = split + 1;
        }

        __global BVHNodeGPU * node = &outNodes[index];
        node->leftChildIndex  = leftIndex;
        node->rightChildIndex = rightIndex;
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void post_build_lbvh(
    __global       BVHNodeGPU * outNodes,
             const uint         N)
{
    const uint index = get_global_id(0);

    if (index >= N) {
        return;
    }

    __global       BVHNodeGPU * node = outNodes + index;
    __global const BVHNodeGPU * left  = outNodes + node->leftChildIndex;
    __global const BVHNodeGPU * right = outNodes + node->rightChildIndex;

    AABBGPU aabb;
    aabb.min_x = min(left->aabb.min_x, right->aabb.min_x);
    aabb.min_y = min(left->aabb.min_y, right->aabb.min_y);
    aabb.min_z = min(left->aabb.min_z, right->aabb.min_z);
    aabb.max_x = max(left->aabb.max_x, right->aabb.max_x);
    aabb.max_y = max(left->aabb.max_y, right->aabb.max_y);
    aabb.max_z = max(left->aabb.max_z, right->aabb.max_z);

    node->aabb = aabb;
}

// =============================================================================
// Merge Sort
// =============================================================================

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const Prim * src,
    __global       Prim * dst,
             const uint   iter,
             const uint   size
) {
    const uint index = get_global_id(0);

    if (index >= size) {
        return;
    }

    const uint len = 1u << (iter - 1);
    const uint left_bound = (index >> iter) << iter;
    const uint middle = left_bound + len;
    const Prim prim = src[index];

    if (middle >= size) {
        dst[index] = prim;
        return;
    }

    uint target = prim.morton;
    uint left, right, const_left, offset;
    if (index < middle) {
        offset = left_bound;
        const_left = left = middle - 1;
        right = min(middle + len, size);
    } else {
        offset = middle;
        const_left = left = left_bound - 1;
        right = middle;
        target += 1;
    }

    while (right - left > 1) {
        const uint mid = (left + right) / 2;
        const uint cmp = src[mid].morton < target;
        const uint ncmp = 1 - cmp;
        left = cmp * mid + ncmp * left;
        right = ncmp * mid + cmp * right;
    }

    const uint dst_index = left_bound + (index - offset) + (left - const_left);
    dst[dst_index] = prim;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void small_merge_sort(
    __global const Prim * global_src,
    __global       Prim * global_dst,
             const uint   size
) {
    const uint global_index = get_global_id(0);

    if (global_index >= size) {
        return;
    }

    __local Prim buf1[GROUP_SIZE];
    __local Prim buf2[GROUP_SIZE];
    __local Prim* src = buf1;
    __local Prim* dst = buf2;

    const uint index = get_local_id(0);
    src[index] = global_src[global_index];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint iter = 1; iter < PIVOT; ++iter) {
        const uint len = 1u << (iter - 1);
        const uint left_bound = (index >> iter) << iter;
        const uint middle = left_bound + len;

        uint target = src[index].morton;
        uint left, right, const_left, offset;
        if (index < middle) {
            offset = left_bound;
            const_left = left = middle - 1;
            right = min(middle + len, (uint)GROUP_SIZE);
        } else {
            offset = middle;
            const_left = left = left_bound - 1;
            right = middle;
            target += 1;
        }

        while (right - left > 1) {
            const uint mid = (left + right) / 2;
            const uint cmp = src[mid].morton < target;
            const uint ncmp = 1 - cmp;
            left = cmp * mid + ncmp * left;
            right = ncmp * mid + cmp * right;
        }

        const uint dst_index = left_bound + (index - offset) + (left - const_left);
        dst[dst_index] = src[index];
        barrier(CLK_LOCAL_MEM_FENCE);

        {
            __local unsigned int* tmp = dst;
            dst = src;
            src = tmp;
        }
    }

    global_dst[global_index] = src[index];
}
