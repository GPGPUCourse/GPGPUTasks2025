
#include "helpers/rassert.cl"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"
#include "../defines.h"

// LBVH algo as it is:
// 1. Map each triangle to some nice point so that points as near to each other as their triangles
// 2. Assign Morton's code to each point ans sort'em all
// 3. For each ind [0..n-2] (inner node) check neighbors and go to the larger prefix side with gallopingm as we are diverging from him really late
// 4. Find first pos where prefixes diverges earlier (or same point) -- other end of segment
// 5. Do galloping to find split as the common prefix with closer child is larger than with the other end
// 6. After building tree itself calculate AABBs bottom-to-top with K iterations of plain "if both children's flags are set then calc myself"

// count leading zeros for 32-bit unsigned
static inline int clz32(uint x)
{
    if (x == 0u) return 32;
#if defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse(&idx, x);
    return 31 - int(idx);
#else
    return __builtin_clz(x);
#endif
}


static inline int common_prefix(
    __global const MortonCode* codes,
                   int n,
                   int i,
                   int j
)
{
    if (j < 0 || j >= n || i < 0 || j >= n) return -1;

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

// Build BV-Hierarchy without calculating AABBs.
// It is required for `codes` array to be sorted.
__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void build_lbvh(
    __global const MortonCode* codes,
    __global BVHNodeGPU* lbvh,
             int n
)
{
    int i = get_global_id(0);
    if (i >= n-1) {
        return;
    }

    int d = (common_prefix(codes, n, i, i-1) < common_prefix(codes, n, i, i+1) ? +1 : -1);
    int pref_min = common_prefix(codes, n, i, i-d);
    int jmp = 1;
    while (n >= jmp * 2) jmp *= 2;
    int init_jmp = jmp;

    int j = i;
    while (jmp > 0) {
        int pref_jmp = common_prefix(codes, n, i, j + d * jmp);
        if (pref_jmp > pref_min) // we are still haven't diverged in BVH-tree with i'th leaf
            j += d * jmp;
        jmp /= 2;
    }
    // now segment with ends in i, j is the i'th node segment
    // Let's find da split
    jmp = init_jmp;
    int pref_ends = common_prefix(codes, n, i, j);
    int split = i;
    while (jmp > 0) {
        if (common_prefix(codes, n, i, split + d * jmp) > pref_ends) {
            split += d * jmp;
        }
        jmp /= 2;
    }
    int split_l = split, split_r = split + d;
    if (split_l > split_r) {
        split_l = split_r;
        split_r = split;
    }

    int left_ind, right_ind;
    // process leafy child if so
    if (min(i, j) == split_l)
        left_ind = (n-1) + split_l;
    else
        left_ind = split_l;
    
    if (max(i, j) == split_r)
        right_ind = (n-1) + split_r;
    else
        right_ind = split_r;
    

    // printf("i=%d j=%d, left=%d right=%d leafStart=%d\n", i, j, left_ind, right_ind, n-1);
    
    lbvh[i].leftChildIndex = left_ind;
    lbvh[i].rightChildIndex = right_ind;
}
