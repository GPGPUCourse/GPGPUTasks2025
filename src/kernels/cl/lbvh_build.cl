#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

static inline int common_prefix(__global const MortonCode* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;

    MortonCode ci = codes[i];
    MortonCode cj = codes[j];

    if (ci == cj) {
        uint diff = i ^ j;
        return 32 + clz(diff);
    } else {
        uint diff = ci ^ cj;
        return clz(diff);
    }
}

static inline void determine_range(__global const MortonCode* codes, int N, int i, __private int* outFirst, __private int* outLast)
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

static inline int find_split(
    __global const MortonCode* codes,
    const int N,
    int first, 
    int last
)
{
    if (first == last)
        return first;

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

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
lbvh_build(
    __global const MortonCode* morton_codes,
    __global BVHNodeGPU*       outNodes,
    __global uint*             parent,
    const uint                 N)
{
    const uint i = get_global_id(0);
    if (i >= N - 1) {
        return;
    }

    int first, last;
    determine_range(morton_codes, N, i, &first, &last);
    int split = find_split(morton_codes, N, first, last);

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

    outNodes[i].leftChildIndex = leftIndex;
    outNodes[i].rightChildIndex = rightIndex;
    parent[leftIndex] = i;
    parent[rightIndex] = i;
}