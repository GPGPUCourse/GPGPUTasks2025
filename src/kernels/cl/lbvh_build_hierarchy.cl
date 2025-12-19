#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

static inline int clz32(uint x)
{
    return x ? clz(x) : 32;
}

static inline int lcp(__global const uint* morton, int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;

    uint a = morton[i];
    uint b = morton[j];

    if (a == b) {
        uint x = (uint)(i ^ j);
        return 32 + clz32(x);
    }
    return clz32(a ^ b);
}

static inline int find_split(__global const uint* morton, int n, int first, int last)
{
    uint first_code = morton[first];
    uint last_code  = morton[last];

    if (first_code == last_code)
        return (first + last) >> 1;

    int common_prefix = clz32(first_code ^ last_code);

    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        int new_split = split + step;
        if (new_split < last) {
            uint split_code = morton[new_split];
            int split_prefix = clz32(first_code ^ split_code);
            if (split_prefix > common_prefix) split = new_split;
        }
    } while (step > 1);

    return split;
}

__kernel void lbvh_build_hierarchy(
    __global const uint* morton_sorted,
    __global BVHNodeGPU* nodes,
    __global int* parent,
    uint nfaces)
{
    int i = (int)get_global_id(0);
    int n = (int)nfaces;
    if (i >= n - 1) return;

    int d = (lcp(morton_sorted, n, i, i + 1) - lcp(morton_sorted, n, i, i - 1)) >= 0 ? 1 : -1;

    int lcp_min = lcp(morton_sorted, n, i, i - d);

    int lmax = 2;
    while (lcp(morton_sorted, n, i, i + lmax * d) > lcp_min)
        lmax <<= 1;

    int l = 0;
    for (int t = lmax >> 1; t > 0; t >>= 1) {
        if (lcp(morton_sorted, n, i, i + (l + t) * d) > lcp_min)
            l += t;
    }

    int j = i + l * d;
    int first = min(i, j);
    int last  = max(i, j);

    int split = find_split(morton_sorted, n, first, last);

    int leafStart = n - 1;

    int leftChild  = (split == first) ? (leafStart + split) : split;
    int rightChild = (split + 1 == last) ? (leafStart + split + 1) : (split + 1);

    nodes[i].leftChildIndex = (uint)leftChild;
    nodes[i].rightChildIndex = (uint)rightChild;

    parent[leftChild] = i;
    parent[rightChild] = i;

    if (i == 0) parent[i] = -1;
}
