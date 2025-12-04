#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"


int common_prefix(__global const MortonCode* codes, int N, int i, int j)
{
    if (j < 0 || j >= N) return -1;

    MortonCode ci = codes[i];
    MortonCode cj = codes[j];

    if (ci == cj) {
        // Use index as tie-breaker
        return 32 + clz((uint)i ^ (uint)j);
    } else {
        return clz(ci ^ cj);
    }
}

void determine_range(__global const MortonCode* codes, int N, int i, int* outFirst, int* outLast)
{
    // direction of the range
    int d = (common_prefix(codes, N, i, i + 1) - common_prefix(codes, N, i, i - 1)) >= 0 ? 1 : -1;

    int deltaMin = common_prefix(codes, N, i, i - d);
    
    int lmax = 2;
    while (common_prefix(codes, N, i, i + lmax * d) > deltaMin) {
        lmax <<= 1;
    }
    
    int l = 0;
    for (int t = lmax / 2; t > 0; t /= 2) {
        if (common_prefix(codes, N, i, i + (l + t) * d) > deltaMin) {
            l += t;
        }
    }
    
    int j = i + l * d;
    *outFirst = min(i, j);
    *outLast = max(i, j);
}


int find_split(__global const MortonCode* codes, int N, int first, int last)
{
    int commonPrefix = common_prefix(codes, N, first, last);
    
    int split = first;
    int step = last - first;
    
    step = (step + 1) >> 1;
    int newSplit = split + step;
    if (newSplit < last) {
        int splitPrefix = common_prefix(codes, N, first, newSplit);
        if (splitPrefix > commonPrefix) {
            split = newSplit;
        }
    }

    while (step > 1) {
        step = (step + 1) >> 1;
        newSplit = split + step;

        if (newSplit < last) {
            int splitPrefix = common_prefix(codes, N, first, newSplit);
            if (splitPrefix > commonPrefix) {
                split = newSplit;
            }
        }
    }
    
    return split;
}

__kernel void lbvh_4_build_internal(
    __global const MortonCode* codes,
    __global BVHNodeGPU* bvh_nodes,
    __global uint* parents,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces - 1) return; // Internal nodes are 0 to N-2

    int first, last;
    determine_range(codes, n_faces, i, &first, &last);
    
    int split = find_split(codes, n_faces, first, last);
    
    int leftIndex;
    int rightIndex;
    
    // left child
    if (split == first) {
        leftIndex = (n_faces - 1) + split; // Leaf
    } else {
        leftIndex = split; // Internal
    }
    
    // right child
    if (split + 1 == last) {
        rightIndex = (n_faces - 1) + split + 1; // Leaf
    } else {
        rightIndex = split + 1; // Internal
    }
    
    bvh_nodes[i].leftChildIndex = leftIndex;
    bvh_nodes[i].rightChildIndex = rightIndex;

    parents[leftIndex] = i;
    parents[rightIndex] = i;

    if (i == 0) {
        parents[0] = 0xFFFFFFFF; // Sentinel
    }
}
