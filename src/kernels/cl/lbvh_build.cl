#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "geometry_helpers.cl"

// Helper: Morton Code

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
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    uint ix = (uint)x;
    uint iy = (uint)y;
    uint iz = (uint)z;
    uint xx = expandBits(ix);
    uint yy = expandBits(iy);
    uint zz = expandBits(iz);
    return (xx << 2) | (yy << 1) | zz;
}

// Kernel: Compute Morton Codes

__kernel void compute_morton_codes(
    __global const float* vertices,
    __global const uint* faces,
    __global const float* scene_aabb_min, // min_x, min_y, min_z
    __global const float* scene_aabb_max, // max_x, max_y, max_z
    __global MortonCode* out_codes,
    __global uint* out_indices,
    uint n)
{
    uint idx = get_global_id(0);
    if (idx >= n) return;

    uint3 f = loadFace(faces, idx);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    float3 centroid = (v0 + v1 + v2) * (1.0f / 3.0f);

    float3 min_p = (float3)(scene_aabb_min[0], scene_aabb_min[1], scene_aabb_min[2]);
    float3 max_p = (float3)(scene_aabb_max[0], scene_aabb_max[1], scene_aabb_max[2]);
    float3 diff = max_p - min_p;
    // Avoid division by zero
    diff.x = max(diff.x, 1e-9f);
    diff.y = max(diff.y, 1e-9f);
    diff.z = max(diff.z, 1e-9f);

    float3 norm = (centroid - min_p) / diff;

    out_codes[idx] = morton3D(norm.x, norm.y, norm.z);
    out_indices[idx] = idx;
}

// Kernel: Radix Sort Scatter Payload

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void radix_sort_scatter_payload(
    __global const uint* values,
    __global const uint* payload,
    __global const uint* offsets,
    __global uint* out_values,
    __global uint* out_payload,
    unsigned int n,
    unsigned int bit)
{
    const uint lid = get_local_id(0);
    const uint gid = get_group_id(0);
    const uint idx = gid * 256 + lid;

    __local uint l_offsets[(1 << 4)];
    __local uint l_buckets[256];

    for (uint i = 0; i < (1 << 4); i += 256) {
        if (lid + i < (1 << 4)) {
            l_offsets[lid + i] = offsets[gid * (1 << 4) + lid + i];
        }
    }

    uint valid = idx < n ? 1u : 0u;
    uint v = 0;
    uint p = 0;
    uint bucket = 0xFFFFFFFF;
    
    if (valid) {
        v = values[idx];
        p = payload[idx];
        bucket = (v >> bit) & 0xF;
    }
    l_buckets[lid] = bucket;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (valid) {
        uint pos = 0;
        for (uint k = 0; k < lid; ++k) {
            if (l_buckets[k] == bucket) {
                pos++;
            }
        }
        
        uint dst_idx = l_offsets[bucket] + pos;
        out_values[dst_idx] = v;
        out_payload[dst_idx] = p;
    }
}

// Kernel: Initialize Leaves

__kernel void init_leaves(
    __global const uint* leafTriIndices,
    __global const float* vertices,
    __global const uint* faces,
    __global BVHNodeGPU* nodes,
    uint n)
{
    uint idx = get_global_id(0);
    if (idx >= n) return;

    uint triIdx = leafTriIndices[idx];
    uint3 f = loadFace(faces, triIdx);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    AABBGPU aabb;
    aabb.min_x = min(min(v0.x, v1.x), v2.x);
    aabb.min_y = min(min(v0.y, v1.y), v2.y);
    aabb.min_z = min(min(v0.z, v1.z), v2.z);
    aabb.max_x = max(max(v0.x, v1.x), v2.x);
    aabb.max_y = max(max(v0.y, v1.y), v2.y);
    aabb.max_z = max(max(v0.z, v1.z), v2.z);

    uint leafNodeIdx = (n - 1) + idx;
    nodes[leafNodeIdx].aabb = aabb;
    nodes[leafNodeIdx].leftChildIndex = 0xFFFFFFFF;
    nodes[leafNodeIdx].rightChildIndex = 0xFFFFFFFF;
}

// Kernel: Build Hierarchy

static inline int common_prefix(__global const MortonCode* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;
    
    MortonCode ci = codes[i];
    MortonCode cj = codes[j];
    
    if (ci == cj) {
        uint diff = (uint)i ^ (uint)j;
        return 32 + clz(diff);
    } else {
        uint diff = ci ^ cj;
        return clz(diff);
    }
}

__kernel void build_internal_nodes(
    __global const MortonCode* codes,
    __global BVHNodeGPU* nodes,
    __global int* parents,
    int N)
{
    int i = get_global_id(0);
    if (i >= N - 1) return;

    int delta_L = common_prefix(codes, N, i, i - 1);
    int delta_R = common_prefix(codes, N, i, i + 1);
    
    int d = (delta_R > delta_L) ? 1 : -1;
    int delta_min = common_prefix(codes, N, i, i - d);
    
    int l_max = 2;
    while (common_prefix(codes, N, i, i + l_max * d) > delta_min) {
        l_max <<= 1;
    }
    
    int l = 0;
    for (int t = l_max >> 1; t > 0; t >>= 1) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > delta_min) {
            l += t;
        }
    }
    
    int j = i + l * d;
    int first = min(i, j);
    int last = max(i, j);
    
    int delta_node = common_prefix(codes, N, first, last);
    int split = first;
    int step = last - first;
    
    do {
        step = (step + 1) >> 1;
        int new_split = split + step;
        if (new_split < last) {
            int delta_split = common_prefix(codes, N, first, new_split);
            if (delta_split > delta_node) {
                split = new_split;
            }
        }
    } while (step > 1);
    
    int leftIdx, rightIdx;
    
    if (split == first) {
        leftIdx = (N - 1) + split;
    } else {
        leftIdx = split;
    }
    
    if (split + 1 == last) {
        rightIdx = (N - 1) + split + 1;
    } else {
        rightIdx = split + 1;
    }
    
    nodes[i].leftChildIndex = leftIdx;
    nodes[i].rightChildIndex = rightIdx;
    
    parents[leftIdx] = i;
    parents[rightIdx] = i;
    
    if (i == 0) {
        parents[i] = -1; // root
    }
}

// Kernel: Compute AABBs

__kernel void compute_aabbs(
    __global BVHNodeGPU* nodes,
    __global const int* parents,
    __global int* counters,
    int N)
{
    int idx = get_global_id(0);
    if (idx >= N) return;
    
    int current = (N - 1) + idx;
    
    while (true) {
        int parent = parents[current];
        if (parent == -1) break;
        
        int old = atomic_inc(&counters[parent]);
        
        if (old == 0) {
            return;
        }
        
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        
        int leftIdx = nodes[parent].leftChildIndex;
        int rightIdx = nodes[parent].rightChildIndex;
        
        AABBGPU leftAABB = nodes[leftIdx].aabb;
        AABBGPU rightAABB = nodes[rightIdx].aabb;
        
        AABBGPU aabb;
        aabb.min_x = min(leftAABB.min_x, rightAABB.min_x);
        aabb.min_y = min(leftAABB.min_y, rightAABB.min_y);
        aabb.min_z = min(leftAABB.min_z, rightAABB.min_z);
        aabb.max_x = max(leftAABB.max_x, rightAABB.max_x);
        aabb.max_y = max(leftAABB.max_y, rightAABB.max_y);
        aabb.max_z = max(leftAABB.max_z, rightAABB.max_z);
        
        nodes[parent].aabb = aabb;
        
        current = parent;
    }
}
