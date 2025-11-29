#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#define INVALID 0xffffffff

int clz32(uint x)
{
    if (x == 0u) return 32;
    return __builtin_clz(x);
}

int common_prefix(__global const MortonCode* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) {
        return -1;
    }

    MortonCode ci = codes[i];
    MortonCode cj = codes[j];

    if (ci == cj) {
        uint di = i;
        uint dj = j;
        uint diff = di ^ dj;
        return 32 + clz32(diff);
    } else {
        uint diff = ci ^ cj;
        return clz32(diff);
    }
}

void determine_range(__global const uint* codes, int N, int i, int* outFirst, int* outLast)
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

int find_split(__global const uint* codes, int N,
                       int first, int last)
{
    if (first == last) {
        return first;
    }

    int commonPrefix = common_prefix(codes, N, first, last);

    int split = first;
    int step  = last - first;

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

__kernel void lbvh_build_internals(
    __global BVHNodeGPU*       bvhNodes,
    __global const uint*       sortedCodes,
    uint                       nfaces)
{
    const uint globalIdx = get_global_id(0);

    for (uint i = globalIdx * BOX_BLOCK_SIZE; i < min((globalIdx + 1) * BOX_BLOCK_SIZE, nfaces - 1); ++i) {
        int first, last;
        determine_range(sortedCodes, nfaces, i, &first, &last);
        int split = find_split(sortedCodes, nfaces, first, last);

        int leftIndex;
        if (split == first) {
            leftIndex = (nfaces - 1) + split;
        } else {
            leftIndex = split;
        }

        int rightIndex;
        if (split + 1 == last) {
            rightIndex = (nfaces - 1) + split + 1;
        } else {
            rightIndex = split + 1;
        }

        BVHNodeGPU node;
        node.leftChildIndex  = leftIndex;
        node.rightChildIndex = rightIndex;
        bvhNodes[i] = node;
    }
}
