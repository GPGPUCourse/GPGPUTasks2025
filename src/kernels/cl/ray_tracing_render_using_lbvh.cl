#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

// ---- Morton helpers -------------------------------------------------------
static inline uint expandBits(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline MortonCode morton3D(float x, float y, float z)
{
    uint ix = (uint)clamp((int)(x * 1024.0f), 0, 1023);
    uint iy = (uint)clamp((int)(y * 1024.0f), 0, 1023);
    uint iz = (uint)clamp((int)(z * 1024.0f), 0, 1023);

    uint xx = expandBits(ix);
    uint yy = expandBits(iy);
    uint zz = expandBits(iz);
    return (xx << 2) | (yy << 1) | zz;
}

// Compute per-triangle Morton key and leaf AABB (unsorted)
__kernel void compute_morton_and_leaf_aabb(
    __global const float* vertices,
    __global const uint* faces,
    __global ulong* morton_keys,
    __global AABBGPU* leaf_aabbs,
    uint nfaces,
    float cmin_x, float cmin_y, float cmin_z,
    float cmax_x, float cmax_y, float cmax_z)
{
    const uint idx = get_global_id(0);
    if (idx >= nfaces)
        return;

    uint3 f = loadFace(faces, idx);
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
    leaf_aabbs[idx] = aabb;

    float3 centroid = (float3)((v0.x + v1.x + v2.x) * (1.0f / 3.0f),
        (v0.y + v1.y + v2.y) * (1.0f / 3.0f),
        (v0.z + v1.z + v2.z) * (1.0f / 3.0f));

    const float eps = 1e-9f;
    float nx = (centroid.x - cmin_x) / fmax(cmax_x - cmin_x, eps);
    float ny = (centroid.y - cmin_y) / fmax(cmax_y - cmin_y, eps);
    float nz = (centroid.z - cmin_z) / fmax(cmax_z - cmin_z, eps);
    nx = clamp(nx, 0.0f, 1.0f);
    ny = clamp(ny, 0.0f, 1.0f);
    nz = clamp(nz, 0.0f, 1.0f);

    uint morton = morton3D(nx, ny, nz);
    morton_keys[idx] = (((ulong)morton) << 32) | (ulong)idx;
}

// Helper for merge sort on 64-bit keys (key packs morton<<32 | triIndex)
static inline uint2 merge_path_partition64(uint diag,
    __global const ulong* input_data,
    uint left_start,
    uint left_len,
    uint right_start,
    uint right_len)
{
    uint low = (diag > right_len) ? diag - right_len : 0u;
    if (low > left_len)
        low = left_len;
    uint high = (diag < left_len) ? diag : left_len;

    while (low < high) {
        uint mid = (low + high) >> 1;
        uint left_idx = mid;
        uint right_idx = diag - mid;

        const int has_left = (int)(left_idx < left_len);
        const int has_right_prev = (int)(right_idx > 0u);

        ulong left_val = 0ul;
        if (has_left)
            left_val = input_data[left_start + left_idx];

        ulong right_prev = 0ul;
        if (has_right_prev)
            right_prev = input_data[right_start + right_idx - 1u];

        const bool move_low = has_left && has_right_prev && (left_val < right_prev);
        if (move_low) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }

    const uint left_idx = low;
    const uint right_idx = diag - left_idx;
    return (uint2)(left_idx, right_idx);
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void
merge_sort64(
    __global const ulong* input_data,
    __global ulong* output_data,
    int sorted_k,
    int n)
{
    if (sorted_k <= 0 || n <= 0)
        return;

    const uint local_id = get_local_id(0);
    const uint group_id = get_group_id(0);
    const uint num_groups = get_num_groups(0);

    const ulong block_size64 = (ulong)sorted_k * 2ul;
    const ulong total_segments = block_size64 == 0ul ? 0ul : ((ulong)n + block_size64 - 1ul) / block_size64;
    if (total_segments == 0ul)
        return;

    for (ulong segment = (ulong)group_id; segment < total_segments; segment += (ulong)num_groups) {
        const ulong block_start64 = block_size64 * segment;
        if (block_start64 >= (ulong)n)
            continue;

        const uint block_start = (uint)block_start64;
        const ulong left_available = (ulong)n - block_start64;
        const uint left_len = (uint)min(left_available, (ulong)sorted_k);
        const ulong right_start64 = block_start64 + (ulong)left_len;
        const ulong right_available = (right_start64 < (ulong)n) ? ((ulong)n - right_start64) : 0ul;
        const uint right_len = (uint)min(right_available, (ulong)sorted_k);
        const uint total_len = left_len + right_len;
        if (total_len == 0u)
            continue;

        const uint right_start = (uint)right_start64;
        const uint chunk = (total_len + 256 - 1u) / 256;
        const uint begin = min(total_len, local_id * chunk);
        if (begin >= total_len)
            continue;

        const uint end = min(total_len, begin + chunk);
        const uint out_offset = block_start + begin;
        const uint iterations = end - begin;

        uint2 start_pair = merge_path_partition64(begin, input_data, block_start, left_len, right_start, right_len);
        uint left_index = start_pair.x;
        uint right_index = start_pair.y;

        const ulong sentinel = 0xfffffffffffffffful;
        for (uint k = 0; k < iterations; ++k) {
            const bool has_left = left_index < left_len;
            const bool has_right = right_index < right_len;
            const ulong left_val = has_left ? input_data[block_start + left_index] : sentinel;
            const ulong right_val = has_right ? input_data[right_start + right_index] : sentinel;

            const bool take_left = has_left && (!has_right || left_val <= right_val);
            output_data[out_offset + k] = take_left ? left_val : right_val;
            left_index += take_left ? 1u : 0u;
            right_index += take_left ? 0u : 1u;
        }
    }
}

// Scatter sorted keys into leaf nodes and outputs
__kernel void scatter_leaves(
    __global const ulong* sorted_keys,
    __global const AABBGPU* leaf_aabbs_unsorted,
    __global BVHNodeGPU* nodes,
    __global uint* leafTriIndices,
    __global uint* sorted_codes,
    __global uint* ready,
    uint nfaces)
{
    const uint idx = get_global_id(0);
    if (idx >= nfaces)
        return;

    const uint leafIndex = (uint)(nfaces - 1u + idx);
    const ulong key = sorted_keys[idx];
    const uint triIndex = (uint)(key & 0xfffffffful);
    const uint morton = (uint)(key >> 32);

    nodes[leafIndex].aabb = leaf_aabbs_unsorted[triIndex];
    nodes[leafIndex].leftChildIndex = 0xffffffffu;
    nodes[leafIndex].rightChildIndex = 0xffffffffu;

    leafTriIndices[idx] = triIndex;
    sorted_codes[idx] = morton;
    ready[leafIndex] = 1u;
}

// Common-prefix helpers for internal node topology construction
static inline int clz32(uint x) { return (int)clz(x); }

static inline int common_prefix(__global const uint* codes, int N, int i, int j)
{
    if (j < 0 || j >= N)
        return -1;
    uint ci = codes[i];
    uint cj = codes[j];
    if (ci == cj) {
        uint diff = (uint)(i ^ j);
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
    *outLast = max(i, j);
}

static inline int find_split(__global const uint* codes, int N, int first, int last)
{
    if (first == last)
        return first;

    int commonPrefix = common_prefix(codes, N, first, last);
    int split = first;
    int step = last - first;
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

// Build internal node children (topology only)
__kernel void build_internal_nodes(
    __global const uint* codes,
    __global BVHNodeGPU* nodes,
    uint nfaces)
{
    const uint idx = get_global_id(0);
    const uint internal_count = (nfaces > 0) ? (nfaces - 1u) : 0u;
    if (idx >= internal_count)
        return;

    int first, last;
    determine_range(codes, (int)nfaces, (int)idx, &first, &last);
    int split = find_split(codes, (int)nfaces, first, last);

    int leftIndex;
    if (split == first) {
        leftIndex = (int)((nfaces - 1u) + split);
    } else {
        leftIndex = split;
    }

    int rightIndex;
    if (split + 1 == last) {
        rightIndex = (int)((nfaces - 1u) + split + 1);
    } else {
        rightIndex = split + 1;
    }

    nodes[idx].leftChildIndex = (uint)leftIndex;
    nodes[idx].rightChildIndex = (uint)rightIndex;
}

// Propagate AABB bottom-up; run iteratively until no changes
__kernel void propagate_aabb_once(
    __global BVHNodeGPU* nodes,
    __global const uint* ready,
    __global uint* ready_out,
    uint nfaces)
{
    const uint idx = get_global_id(0);
    const uint internal_count = (nfaces > 0) ? (nfaces - 1u) : 0u;
    if (idx >= internal_count)
        return;

    // preserve previous readiness
    ready_out[idx] = ready[idx];
    if (ready[idx])
        return;

    const uint left = nodes[idx].leftChildIndex;
    const uint right = nodes[idx].rightChildIndex;
    if (ready[left] && ready[right]) {
        AABBGPU aabb;
        const AABBGPU leftA = nodes[left].aabb;
        const AABBGPU rightA = nodes[right].aabb;
        aabb.min_x = fmin(leftA.min_x, rightA.min_x);
        aabb.min_y = fmin(leftA.min_y, rightA.min_y);
        aabb.min_z = fmin(leftA.min_z, rightA.min_z);
        aabb.max_x = fmax(leftA.max_x, rightA.max_x);
        aabb.max_y = fmax(leftA.max_y, rightA.max_y);
        aabb.max_z = fmax(leftA.max_z, rightA.max_z);
        nodes[idx].aabb = aabb;
        ready_out[idx] = 1u;
    }
}

// BVH traversal: closest hit along ray
static inline bool bvh_closest_hit(
    const float3 orig,
    const float3 dir,
    __global const BVHNodeGPU* nodes,
    __global const uint* leafTriIndices,
    uint nfaces,
    __global const float* vertices,
    __global const uint* faces,
    float tMin,
    __private float* outT, // сюда нужно записать t рассчитанный в intersect_ray_triangle(..., t, u, v)
    __private int* outFaceId,
    __private float* outU, // сюда нужно записать u рассчитанный в intersect_ray_triangle(..., t, u, v)
    __private float* outV) // сюда нужно записать v рассчитанный в intersect_ray_triangle(..., t, u, v)
{
    const int MAX_STACK = 64; // depth bound
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1;

    // iterative DFS with explicit stack
    int stack[MAX_STACK];
    int stackSize = 0;
    bool hit = false;
    float tBestLocal = *outT;

    stack[stackSize++] = rootIndex;

    while (stackSize > 0) {
        int nodeIndex = stack[--stackSize];
        const BVHNodeGPU node = nodes[nodeIndex];

        float tNear, tFar;
        if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, tBestLocal, &tNear, &tFar))
            continue;

        if (nodeIndex >= leafStart) {
            uint leafId = (uint)(nodeIndex - leafStart);
            uint triIndex = leafTriIndices[leafId];

            uint3 f = loadFace(faces, triIndex);
            float3 a = loadVertex(vertices, f.x);
            float3 b = loadVertex(vertices, f.y);
            float3 c = loadVertex(vertices, f.z);

            float t, u, v;
            if (intersect_ray_triangle(orig, dir,
                    a, b, c,
                    tMin, tBestLocal,
                    false,
                    &t, &u, &v)) {
                hit = true;
                tBestLocal = t;
                *outT = t;
                *outFaceId = (int)triIndex;
                *outU = u;
                *outV = v;
            }
        } else {
            int left = (int)node.leftChildIndex;
            int right = (int)node.rightChildIndex;

            float tNearL, tFarL, tNearR, tFarR;
            bool hitL = intersect_ray_aabb(orig, dir, nodes[left].aabb, tMin, tBestLocal, &tNearL, &tFarL);
            bool hitR = intersect_ray_aabb(orig, dir, nodes[right].aabb, tMin, tBestLocal, &tNearR, &tFarR);

            if (hitL && hitR) {
                // push farther first so nearer is processed next
                if (tNearL < tNearR) {
                    if (stackSize < MAX_STACK)
                        stack[stackSize++] = right;
                    if (stackSize < MAX_STACK)
                        stack[stackSize++] = left;
                } else {
                    if (stackSize < MAX_STACK)
                        stack[stackSize++] = left;
                    if (stackSize < MAX_STACK)
                        stack[stackSize++] = right;
                }
            } else if (hitL) {
                if (stackSize < MAX_STACK)
                    stack[stackSize++] = left;
            } else if (hitR) {
                if (stackSize < MAX_STACK)
                    stack[stackSize++] = right;
            }
        }
    }

    return hit;
}

// Cast a single ray and report if ANY occluder is hit (for ambient occlusion)
static inline bool any_hit_from(
    const float3 orig,
    const float3 dir,
    __global const float* vertices,
    __global const uint* faces,
    __global const BVHNodeGPU* nodes,
    __global const uint* leafTriIndices,
    uint nfaces,
    int ignore_face)
{
    const int MAX_STACK = 64;
    const int rootIndex = 0;
    const int leafStart = (int)nfaces - 1;

    const float tMin = 1e-4f;

    int stack[MAX_STACK];
    int stackSize = 0;
    stack[stackSize++] = rootIndex;

    while (stackSize > 0) {
        int nodeIndex = stack[--stackSize];
        const BVHNodeGPU node = nodes[nodeIndex];

        float tNear, tFar;
        if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, FLT_MAX, &tNear, &tFar))
            continue;

        if (nodeIndex >= leafStart) {
            uint leafId = (uint)(nodeIndex - leafStart);
            uint triIndex = leafTriIndices[leafId];
            if ((int)triIndex == ignore_face)
                continue;

            uint3 f = loadFace(faces, triIndex);
            float3 a = loadVertex(vertices, f.x);
            float3 b = loadVertex(vertices, f.y);
            float3 c = loadVertex(vertices, f.z);

            float t, u, v;
            if (intersect_ray_triangle(orig, dir,
                    a, b, c,
                    tMin, FLT_MAX,
                    false,
                    &t, &u, &v)) {
                return true;
            }
        } else {
            int left = (int)node.leftChildIndex;
            int right = (int)node.rightChildIndex;

            float tNearL, tFarL, tNearR, tFarR;
            bool hitL = intersect_ray_aabb(orig, dir, nodes[left].aabb, tMin, FLT_MAX, &tNearL, &tFarL);
            bool hitR = intersect_ray_aabb(orig, dir, nodes[right].aabb, tMin, FLT_MAX, &tNearR, &tFarR);

            if (hitL && hitR) {
                if (tNearL < tNearR) {
                    if (stackSize < MAX_STACK)
                        stack[stackSize++] = right;
                    if (stackSize < MAX_STACK)
                        stack[stackSize++] = left;
                } else {
                    if (stackSize < MAX_STACK)
                        stack[stackSize++] = left;
                    if (stackSize < MAX_STACK)
                        stack[stackSize++] = right;
                }
            } else if (hitL) {
                if (stackSize < MAX_STACK)
                    stack[stackSize++] = left;
            } else if (hitR) {
                if (stackSize < MAX_STACK)
                    stack[stackSize++] = right;
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
    __global const float* vertices,
    __global const uint* faces,
    __global const BVHNodeGPU* bvhNodes,
    __global const uint* leafTriIndices,
    __global int* framebuffer_face_id,
    __global float* framebuffer_ambient_occlusion,
    __global const CameraViewGPU* camera,
    uint nfaces)
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

    float tMin = 1e-6f;
    float tBest = FLT_MAX;
    float uBest = 0.0f;
    float vBest = 0.0f;
    int faceIdBest = -1;

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
        uint3 f = loadFace(faces, (uint)faceIdBest);
        float3 a = loadVertex(vertices, f.x);
        float3 b = loadVertex(vertices, f.y);
        float3 c = loadVertex(vertices, f.z);

        float3 e1 = (float3)(b.x - a.x, b.y - a.y, b.z - a.z);
        float3 e2 = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float3 n = normalize_f3(cross_f3(e1, e2));

        // ensure hemisphere is "outside" relative to the camera ray
        if (n.x * ray_direction.x + n.y * ray_direction.y + n.z * ray_direction.z > 0.0f) {
            n = (float3)(-n.x, -n.y, -n.z);
        }

        float3 P = (float3)(ray_origin.x + tBest * ray_direction.x,
            ray_origin.y + tBest * ray_direction.y,
            ray_origin.z + tBest * ray_direction.z);

        float3 ac = (float3)(c.x - a.x, c.y - a.y, c.z - a.z);
        float scale = fmax(fmax(length_f3(e1), length_f3(e2)),
            length_f3(ac));

        float eps = 1e-3f * fmax(1.0f, scale);
        float3 Po = (float3)(P.x + n.x * eps,
            P.y + n.y * eps,
            P.z + n.z * eps);

        // build tangent basis
        float3 T;
        float3 B;
        make_basis(n, &T, &B);

        // per-pixel seed (stable)
        union {
            float f32;
            uint u32;
        } tBestUnion;
        tBestUnion.f32 = tBest;
        uint rng = (uint)(0x9E3779B9u ^ idx ^ tBestUnion.u32);

        int hits = 0;
        for (int s = 0; s < AO_SAMPLES; ++s) {
            // uniform hemisphere sampling (solid angle)
            float u1 = random01(&rng);
            float u2 = random01(&rng);
            float z = u1; // z in [0,1]
            float phi = 6.28318530718f * u2; // 2*pi*u2
            float r = sqrt(fmax(0.0f, 1.0f - z * z));
            float3 d_local = (float3)(r * cos(phi),
                r * sin(phi),
                z);

            // transform to world space
            float3 d = (float3)(T.x * d_local.x + B.x * d_local.y + n.x * d_local.z,
                T.y * d_local.x + B.y * d_local.y + n.y * d_local.z,
                T.z * d_local.x + B.z * d_local.y + n.z * d_local.z);

            if (any_hit_from(Po, d,
                    vertices, faces,
                    bvhNodes, leafTriIndices,
                    nfaces, faceIdBest)) {
                ++hits;
            }
        }

        ao = 1.0f - (float)hits / (float)AO_SAMPLES; // [0,1]
    }

    framebuffer_ambient_occlusion[idx] = ao;
}
