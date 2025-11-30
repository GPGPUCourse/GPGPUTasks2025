#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

#include "geometry_helpers.cl"

static inline void calculateAABB(
    __global AABBGPU* aabb,
    float3 v0, float3 v1, float3 v2)
{
    aabb->min_x = min(v0.x, min(v1.x, v2.x));
    aabb->min_y = min(v0.y, min(v1.y, v2.y));
    aabb->min_z = min(v0.z, min(v1.z, v2.z));
    aabb->max_x = max(v0.x, max(v1.x, v2.x));
    aabb->max_y = max(v0.y, max(v1.y, v2.y));
    aabb->max_z = max(v0.z, max(v1.z, v2.z));
}

static inline void populateLeaf(
    __global const float* vertices,
    __global const uint* faces,
    __global BVHNodeGPU* bvhNodes,
    uint fi,
    uint index,
    uint leafStart)
{
    uint3 face = loadFace(faces, fi);
    float3 v0 = loadVertex(vertices, face.x);
    float3 v1 = loadVertex(vertices, face.y);
    float3 v2 = loadVertex(vertices, face.z);

    uint node_index = leafStart + index;
    calculateAABB(&bvhNodes[node_index].aabb, v0, v1, v2);

    const uint invalid = UINT_MAX;
    bvhNodes[node_index].leftChildIndex = invalid;
    bvhNodes[node_index].rightChildIndex = invalid;
}

static inline int commonPrefixLen(__global const uint* morton_codes, int i, int j, int nfaces)
{
    if (j < 0 || j >= nfaces) {
        return -1;
    }

    uint code_i = morton_codes[i];
    uint code_j = morton_codes[j];
    if (code_i == code_j) {
        uint x = (uint)(i ^ j);
        return 32 + clz(x);
    } else {
        uint x = code_i ^ code_j;
        return clz(x);
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
build_lbvh_skeleton(
    __global const float* vertices,
    __global const uint* faces,
    __global const uint* morton_codes,
    __global BVHNodeGPU* bvhNodes,
    __global uint* leafTriIndices,
    __global uint* parentIndices,
    uint nfaces)
{
    const uint index = get_global_id(0);

    if (index >= nfaces) {
        return;
    }

    const int leafStart = nfaces - 1;

    populateLeaf(vertices, faces, bvhNodes, leafTriIndices[index], index, leafStart);

    if (index >= leafStart) {
        return;
    }

    int i = index;

    // 1. Range
    int cpL = commonPrefixLen(morton_codes, i, i - 1, nfaces);
    int cpR = commonPrefixLen(morton_codes, i, i + 1, nfaces);
    int dir = (cpR > cpL) ? 1 : -1;

    int deltaMin = commonPrefixLen(morton_codes, i, i - dir, nfaces);
    int lmax = 2;
    while (commonPrefixLen(morton_codes, i, i + lmax * dir, nfaces) > deltaMin) {
        lmax <<= 1;
    }

    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (commonPrefixLen(morton_codes, i, i + (l + t) * dir, nfaces) > deltaMin) {
            l += t;
        }
    }

    int j = i + l * dir;
    int first = min(i, j);
    int last = max(i, j);

    // 2. Split
    int commonPrefix = commonPrefixLen(morton_codes, first, last, nfaces);
    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = commonPrefixLen(morton_codes, first, newSplit, nfaces);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    } while (step > 1);

    // 3. Set childs
    int leftChild, rightChild;

    if (split == first) {
        leftChild = leafStart + split;
    } else {
        leftChild = split;
    }

    if (split + 1 == last) {
        rightChild = leafStart + split + 1;
    } else {
        rightChild = split + 1;
    }

    bvhNodes[index].leftChildIndex = leftChild;
    bvhNodes[index].rightChildIndex = rightChild;
    parentIndices[leftChild] = index;
    parentIndices[rightChild] = index;
}
