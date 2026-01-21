#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

static inline int clz32(uint x) {
    return x == 0 ? 32 : clz(x);
}

static inline int delta(
    __global const MortonCode* codes,
    int n, int i, int j)
{
    if (j < 0 || j >= n) return -1;

    MortonCode ci = codes[i];
    MortonCode cj = codes[j];

    if (ci == cj) {
        return 32 + clz32((uint)i ^ (uint)j);
    } else {
        return clz32(ci ^ cj);
    }
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void lbvh_build_nodes(
    __global const MortonCode* sorted_morton_codes,
    __global const uint* sorted_indices,
    __global BVHNodeGPU* nodes,
    __global uint* leaf_triangle_indices,
    uint nfaces)
{
    uint i = get_global_id(0);
    int n = (int)nfaces;

    if (i >= nfaces) return;

    if (i < nfaces - 1) {
        int d_left = delta(sorted_morton_codes, n, i, i-1);
        int d_right = delta(sorted_morton_codes, n, i, i+1);

        int d = (d_right > d_left) ? 1 : -1;

        int delta_min = delta(sorted_morton_codes, n, i, i - d);
        int lmax = 2;
        while (delta(sorted_morton_codes, n, i, i + lmax * d) > delta_min) {
            lmax *= 2;
        }

        int l = 0;
        for (int t = lmax / 2; t > 0; t /= 2) {
            if (delta(sorted_morton_codes, n, i, i + (l + t) * d) > delta_min) {
                l += t;
            }
        }

        int j = i + l * d;
        int first = min((int)i, j);
        int last = max((int)i, j);

        int common_prefix = delta(sorted_morton_codes, n, first, last);
        int split = first;
        int step = last - first;

        do {
            step = (step + 1) >> 1;
            int new_split = split + step;
            if (new_split < last) {
                int split_prefix = delta(sorted_morton_codes, n, first, new_split);
                if (split_prefix > common_prefix) {
                    split = new_split;
                }
            }
        } while (step > 1);

        int leaf_start = (int)nfaces - 1;
        int left_child = (split == first) ? (leaf_start + split) : split;
        int right_child = (split + 1 == last) ? (leaf_start + split + 1) : (split + 1);

        nodes[i].leftChildIndex = left_child;
        nodes[i].rightChildIndex = right_child;
    }

    int leaf_idx = (int)nfaces - 1 + i;
    leaf_triangle_indices[i] = sorted_indices[i];
    nodes[leaf_idx].leftChildIndex = 0xFFFFFFFFu;
    nodes[leaf_idx].rightChildIndex = 0xFFFFFFFFu;
}
