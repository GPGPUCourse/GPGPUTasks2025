#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "geometry_helpers.cl"

static inline uint expand_bits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static inline uint morton3D(float x, float y, float z) {
    x = clamp(x * 1024.0f, 0.0f, 1023.0f);
    y = clamp(y * 1024.0f, 0.0f, 1023.0f);
    z = clamp(z * 1024.0f, 0.0f, 1023.0f);

    uint xx = expand_bits((uint)x);
    uint yy = expand_bits((uint)y);
    uint zz = expand_bits((uint)z);

    return (xx << 2) | (yy << 1) | zz;
}

__kernel void compute_morton_codes(
    __global const float* vertices,
    __global const uint* faces,
    float sceneMinX, float sceneMinY, float sceneMinZ,
    float sceneMaxX, float sceneMaxY, float sceneMaxZ,
    __global MortonCode* mortonCodes,
    __global uint* triIndices,
    uint N)
{
    uint i = get_global_id(0);
    if (i >= N) return;

    uint3 f = loadFace(faces, i);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    float3 c = (float3)(
        (v0.x + v1.x + v2.x) * (1.0f / 3.0f),
        (v0.y + v1.y + v2.y) * (1.0f / 3.0f),
        (v0.z + v1.z + v2.z) * (1.0f / 3.0f)
    );
    float px = (c.x - sceneMinX) / fmax(sceneMaxX - sceneMinX, 1e-15f);
    float py = (c.y - sceneMinY) / fmax(sceneMaxY - sceneMinY, 1e-15f);
    float pz = (c.z - sceneMinZ) / fmax(sceneMaxZ - sceneMinZ, 1e-15f);
    mortonCodes[i] = morton3D(px, py, pz);
    triIndices[i] = i;
}

__kernel void build_leaf_nodes(
    __global const float* vertices,
    __global const uint*  faces,
    __global const uint*  sortedTriIdx,
    __global BVHNodeGPU*  bvhNodes,
    uint N)
{
    uint global_id = get_global_id(0);
    if (global_id >= N) return;

    uint leafIndex = N - 1 + global_id;
    uint triId = sortedTriIdx[global_id];
    uint3 f = loadFace(faces, triId);
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

    bvhNodes[leafIndex].aabb = aabb;
    bvhNodes[leafIndex].leftChildIndex  = 0xFFFFFFFFu;
    bvhNodes[leafIndex].rightChildIndex = 0xFFFFFFFFu;
}


static inline int common_prefix(__global const MortonCode* codes, int N, int i, int j) {
    if (j < 0 || j >= N) return -1;
    MortonCode ci = codes[i];
    MortonCode cj = codes[j];
    if (ci == cj) {
        uint di = (uint)i;
        uint dj = (uint)j;
        uint diff = di ^ dj;
        return 32 + clz(diff);
    } else {
        uint diff = ci ^ cj;
        return clz(diff);
    }
}

__kernel void build_internal_nodes(
    __global const MortonCode* morton,
    __global BVHNodeGPU* bvhNodes,
    __global uint* parent,
    uint N)
{
    int global_id = get_global_id(0);
    if (global_id >= N - 1) return;

    int cpL = common_prefix(morton, N, global_id, global_id - 1);
    int cpR = common_prefix(morton, N, global_id, global_id + 1);
    int d = cpR >= cpL ? 1: - 1;

    int deltaMin = common_prefix(morton, N, global_id, global_id - d);
    int lmax = 2;
    while (true) {
        int idx = global_id + lmax * d;
        if (idx < 0 || idx >= N) break;
        int cp = common_prefix(morton, N, global_id, idx);
        if (cp > deltaMin) {
            lmax <<= 1;
            continue;
        } else break;
    }

    int z = 0;
    int t = lmax >> 1;
    while (t > 0) {
        if (global_id + (z + t) * d >= 0 && global_id + (z+t) * d < N) {
            int cp = common_prefix(morton, N, global_id, global_id + (z + t) * d);
            if (cp > deltaMin) z += t;
        }
        t >>= 1;
    }

    int first = min(global_id, global_id + z * d);
    int last = max(global_id, global_id + z * d);
    int commonPref = common_prefix(morton, N, first, last);
    int split = first;
    int step = last - first;
    while (step > 1) {
        step = (step + 1) >> 1;
        if (split + step < last) {
            int splitPref = common_prefix(morton, N, first, split + step);
            if (splitPref > commonPref) {
                split = split + step;
            }
        }
    }

    int leftIndex;
    if (split == first) {
        leftIndex = (N - 1) + split;
    } else {
        leftIndex = split;
    }

    int rightIndex;
    if (split + 1 == last) {
        rightIndex = (N - 1) + split + 1;
    } else {
        rightIndex = split + 1;
    }

    BVHNodeGPU node;
    node.leftChildIndex = leftIndex;
    node.rightChildIndex = rightIndex;
    bvhNodes[global_id] = node;

    parent[leftIndex]  = global_id;
    parent[rightIndex] = global_id;
}

__kernel void propagate_aabbs_upwards_from_leaves(
    __global BVHNodeGPU* bvhNodes,
    __global const uint* parent,
    __global uint*       counters,
    uint leafStart,
    uint nLeaves)
{
    uint global_id = get_global_id(0);
    if (global_id >= nLeaves) return;

    uint node = leafStart + global_id;
    while (true) {
        uint p = parent[node];
        if (p == 0xFFFFFFFFu) break;
        uint old = atomic_inc(&counters[p]);
        if (old == 1u) {
            BVHNodeGPU left = bvhNodes[bvhNodes[p].leftChildIndex];
            BVHNodeGPU right = bvhNodes[bvhNodes[p].rightChildIndex];

            AABBGPU merged;
            merged.min_x = fmin(left.aabb.min_x, right.aabb.min_x);
            merged.min_y = fmin(left.aabb.min_y, right.aabb.min_y);
            merged.min_z = fmin(left.aabb.min_z, right.aabb.min_z);
            merged.max_x = fmax(left.aabb.max_x, right.aabb.max_x);
            merged.max_y = fmax(left.aabb.max_y, right.aabb.max_y);
            merged.max_z = fmax(left.aabb.max_z, right.aabb.max_z);

            bvhNodes[p].aabb = merged;
            node = p;
        } else break;
    }
}