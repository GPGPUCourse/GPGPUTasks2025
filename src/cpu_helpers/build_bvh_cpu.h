#pragma once

#include "../kernels/shared_structs/aabb_gpu_shared.h"
#include "../kernels/shared_structs/bvh_node_gpu_shared.h"
#include "morton_code_cpu.h"

#include <libbase/point.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Костыль, так как сборка почему-то сломалась после добавления кернела
#define GPUC_UINT uint32_t

// count leading zeros for 32-bit unsigned
static inline int clz32(uint32_t x)
{
    if (x == 0u)
        return 32;
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse(&idx, x);
    return 31 - static_cast<int>(idx);
#else
    return __builtin_clz(x);
#endif
}

// Compute common prefix length between sorted Morton codes at indices i and j
// Uses index as a tiebreaker when codes are equal (like Karras 2013).
static inline int common_prefix(const std::vector<MortonCode>& codes, int N, int i, int j)
{
    if (j < 0 || j >= N)
        return -1;

    MortonCode ci = codes[static_cast<size_t>(i)];
    MortonCode cj = codes[static_cast<size_t>(j)];

    if (ci == cj) {
        uint32_t di = static_cast<uint32_t>(i);
        uint32_t dj = static_cast<uint32_t>(j);
        uint32_t diff = di ^ dj;
        return 32 + clz32(diff);
    } else {
        uint32_t diff = ci ^ cj;
        return clz32(diff);
    }
}

// Determine range [first, last] of primitives covered by internal node i
static inline void determine_range(const std::vector<uint32_t>& codes, int N, int i, int& outFirst, int& outLast)
{
    int cpL = common_prefix(codes, N, i, i - 1);
    int cpR = common_prefix(codes, N, i, i + 1);

    // Direction of the range
    int d = (cpR > cpL) ? 1 : -1;

    // Find upper bound on the length of the range
    int deltaMin = common_prefix(codes, N, i, i - d);
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

    int j = i + l * d;
    outFirst = std::min(i, j);
    outLast = std::max(i, j);
}

// Find split position inside range [first, last] using the same
// prefix metric as determine_range (code + index tie-break)
static inline int find_split(const std::vector<uint32_t>& codes,
    int first, int last)
{
    const int N = static_cast<int>(codes.size());

    // Degenerate case should not случаться в нормальном коде, но на всякий пожарный
    if (first == last)
        return first;

    auto delta = [&](int i, int j) -> int {
        return common_prefix(codes, N, i, j);
    };

    // Prefix between first and last (уже с учётом индекса, если коды равны)
    int commonPrefix = delta(first, last);

    int split = first;
    int step = last - first;

    // Binary search for the last index < last where
    // prefix(first, i) > prefix(first, last)
    do {
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = delta(first, newSplit);
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
inline void buildLBVH_CPU(
    const std::vector<point3f>& vertices,
    const std::vector<point3u>& faces,
    std::vector<BVHNodeGPU>& outNodes,
    std::vector<uint32_t>& outLeafTriIndices)
{
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
        aabb.min_x = std::min({ v0.x, v1.x, v2.x });
        aabb.min_y = std::min({ v0.y, v1.y, v2.y });
        aabb.min_z = std::min({ v0.z, v1.z, v2.z });
        aabb.max_x = std::max({ v0.x, v1.x, v2.x });
        aabb.max_y = std::max({ v0.y, v1.y, v2.y });
        aabb.max_z = std::max({ v0.z, v1.z, v2.z });

        BVHNodeGPU& node = outNodes[0];
        node.aabb = aabb;
        // Leaf: no children (user can detect leaf via index and N)
        node.leftChildIndex = std::numeric_limits<GPUC_UINT>::max();
        node.rightChildIndex = std::numeric_limits<GPUC_UINT>::max();

        outLeafTriIndices[0] = 0;
        return;
    }

    // Per-triangle info
    struct Prim {
        uint32_t triIndex;
        uint32_t morton;
        AABBGPU aabb;
        point3f centroid;
    };

    std::vector<Prim> prims(N);

    // 1) Compute per-triangle AABB and centroids
    point3f cMin { +std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity(),
        +std::numeric_limits<float>::infinity() };
    point3f cMax { -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity() };

    for (size_t i = 0; i < N; ++i) {
        const point3u& f = faces[i];
        const point3f& v0 = vertices[f.x];
        const point3f& v1 = vertices[f.y];
        const point3f& v2 = vertices[f.z];

        // Triangle AABB
        AABBGPU aabb;
        aabb.min_x = std::min({ v0.x, v1.x, v2.x });
        aabb.min_y = std::min({ v0.y, v1.y, v2.y });
        aabb.min_z = std::min({ v0.z, v1.z, v2.z });
        aabb.max_x = std::max({ v0.x, v1.x, v2.x });
        aabb.max_y = std::max({ v0.y, v1.y, v2.y });
        aabb.max_z = std::max({ v0.z, v1.z, v2.z });

        // Centroid
        point3f c;
        c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
        c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
        c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);

        prims[i].triIndex = static_cast<uint32_t>(i);
        prims[i].aabb = aabb;
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

    // 3) Sort primitives by Morton code
    std::sort(prims.begin(), prims.end(),
        [](const Prim& a, const Prim& b) {
            return a.morton < b.morton;
        });

    // 4) Prepare arrays
    const size_t numNodes = 2 * N - 1;
    outNodes.resize(numNodes);
    outLeafTriIndices.resize(N);

    std::vector<uint32_t> sortedCodes(N);
    for (size_t i = 0; i < N; ++i) {
        sortedCodes[i] = prims[i].morton;
        outLeafTriIndices[i] = prims[i].triIndex;
    }

    // 5) Initialize leaf nodes [N-1 .. 2*N-2]
    const GPUC_UINT INVALID = std::numeric_limits<GPUC_UINT>::max();

    for (size_t i = 0; i < N; ++i) {
        size_t leafIndex = (N - 1) + i;
        BVHNodeGPU& leaf = outNodes[leafIndex];

        leaf.aabb = prims[i].aabb;
        leaf.leftChildIndex = INVALID;
        leaf.rightChildIndex = INVALID;
    }

    // 6) Build internal nodes [0 .. N-2]
    for (int i = 0; i < static_cast<int>(N) - 1; ++i) {
        int first, last;
        determine_range(sortedCodes, static_cast<int>(N), i, first, last);
        int split = find_split(sortedCodes, first, last);

        // Left child
        int leftIndex;
        if (split == first) {
            // Range [first, split] has one primitive -> leaf
            leftIndex = static_cast<int>((N - 1) + split);
        } else {
            // Internal node
            leftIndex = split;
        }

        // Right child
        int rightIndex;
        if (split + 1 == last) {
            // Range [split+1, last] has one primitive -> leaf
            rightIndex = static_cast<int>((N - 1) + split + 1);
        } else {
            // Internal node
            rightIndex = split + 1;
        }

        BVHNodeGPU& node = outNodes[static_cast<size_t>(i)];
        node.leftChildIndex = static_cast<GPUC_UINT>(leftIndex);
        node.rightChildIndex = static_cast<GPUC_UINT>(rightIndex);
    }

    // 7.1) Enumerating bottom-up traversal order for internal nodes to build AABB
    std::vector<int> traversalOrder = { 0 }; // starting with root node
    std::set<int> alreadyVisited;
    size_t next = 0;
    while (next < traversalOrder.size()) {
        int i = traversalOrder[next++];
        rassert(alreadyVisited.count(i) == 0, 452341233412, i, next);
        alreadyVisited.insert(i);
        BVHNodeGPU& node = outNodes[static_cast<size_t>(i)];
        if (node.leftChildIndex != INVALID && node.leftChildIndex < N - 1) { // ensuring it is not a leaf
            traversalOrder.push_back(node.leftChildIndex);
        }
        if (node.rightChildIndex != INVALID && node.rightChildIndex < N - 1) {
            traversalOrder.push_back(node.rightChildIndex);
        }
    }
    rassert(traversalOrder.size() == N - 1, 354623412341, traversalOrder.size(), N);

    // 7.2) AABB propagation for internal nodes in the bottom-up traversal order
    for (int j = traversalOrder.size() - 1; j >= 0; --j) {
        int i = traversalOrder[j];
        rassert(i < outNodes.size(), 45123413211);
        BVHNodeGPU& node = outNodes[static_cast<size_t>(i)];
        const BVHNodeGPU& left = outNodes[static_cast<size_t>(node.leftChildIndex)];
        const BVHNodeGPU& right = outNodes[static_cast<size_t>(node.rightChildIndex)];

        AABBGPU aabb;
        aabb.min_x = std::min(left.aabb.min_x, right.aabb.min_x);
        aabb.min_y = std::min(left.aabb.min_y, right.aabb.min_y);
        aabb.min_z = std::min(left.aabb.min_z, right.aabb.min_z);
        aabb.max_x = std::max(left.aabb.max_x, right.aabb.max_x);
        aabb.max_y = std::max(left.aabb.max_y, right.aabb.max_y);
        aabb.max_z = std::max(left.aabb.max_z, right.aabb.max_z);

        node.aabb = aabb;
    }
}
