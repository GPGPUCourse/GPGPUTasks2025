#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "geometry_helpers.cl"

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
