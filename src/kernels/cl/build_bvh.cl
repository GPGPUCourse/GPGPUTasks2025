#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

inline int match(__global const MortonCode* codes, const MortonCode a,
    const int idx_a, const int idx_b, const uint nfaces)
{
    if (idx_b < 0 || idx_b >= nfaces) {
        return -1;
    }
    uint xor_val = a ^ codes[idx_b];
    if (xor_val == 0) {
        return 32 + clz(idx_a ^ idx_b);
    }
    return clz(xor_val);
}

// TODO try sorted AABB
// TODO try arrays of structures
__kernel void build_bvh(
    __global const MortonCode* sortedCodes,
    const uint nfaces,
    __global BVHNodeGPU* bvhNodes,
    __global uint* parents,
    __global uint* counters)
{
    const int i = (int)get_global_id(0);
    if (i >= nfaces - 1) {
        return;
    }
    
    counters[i] = 0;

    int d = 1; // 1 - right, -1 - left
    MortonCode code = sortedCodes[i];
    int minMatch;
    const int rightMatch = match(sortedCodes, code, i, i + 1, nfaces);
    const int leftMatch = match(sortedCodes, code, i, i - 1, nfaces);
    if (leftMatch >= rightMatch) {
        d = -1;
        minMatch = rightMatch;
    } else {
        minMatch = leftMatch;
    }

    int lmax = 2;
    while (match(sortedCodes, code, i, i + lmax * d, nfaces) > minMatch) {
        lmax <<= 1;
    }

    int l = 0;
    int border = i;
    for (int shift = lmax >> 1; shift > 0; shift >>= 1) {
        int idx = border + shift * d;
        if (match(sortedCodes, code, i, idx, nfaces) > minMatch) {
            border = idx;
        }
    }
    int leftBorder = min(i, border), rightBorder = max(i, border); // borders of current node

    const MortonCode leftCode = sortedCodes[leftBorder];
    const int delta = match(sortedCodes, leftCode, leftBorder, rightBorder, nfaces);
    int split = leftBorder;
    int step = rightBorder - leftBorder;
    int cnt = 0;
    do {
        ++cnt;
        step = (step + 1) >> 1;
        int newSplit = split + step;

        if (newSplit < rightBorder) {
            int splitPrefix = match(sortedCodes, leftCode, leftBorder, newSplit, nfaces);
            if (splitPrefix > delta) {
                split = newSplit;
            }
        }
    } while (step > 1);

    int child = ((split == leftBorder) ? (split + (int)nfaces - 1) : split);
    bvhNodes[i].leftChildIndex = child;
    parents[child] = i;

    ++split;
    child = ((split == rightBorder) ? (split + (int)nfaces - 1) : split);
    bvhNodes[i].rightChildIndex = child;
    parents[child] = i;
}