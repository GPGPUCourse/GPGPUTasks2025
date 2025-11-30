#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

#include "geometry_helpers.cl"

static inline void aabb_union(
    const __global AABBGPU* a,
    const __global AABBGPU* b,
    __global AABBGPU* out)
{
    out->min_x = min(a->min_x, b->min_x);
    out->min_y = min(a->min_y, b->min_y);
    out->min_z = min(a->min_z, b->min_z);
    out->max_x = max(a->max_x, b->max_x);
    out->max_y = max(a->max_y, b->max_y);
    out->max_z = max(a->max_z, b->max_z);
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
populate_aabb_over_lbvh(
    __global BVHNodeGPU* bvhNodes,
    __global const uint* parentIndices,
    __global uint* nodeCounter,
    uint nfaces)
{
    const uint index = get_global_id(0);
    if (index >= nfaces) {
        return;
    }

    const uint leafStart = nfaces - 1;
    uint parent = parentIndices[leafStart + index];

    while (parent != UINT_MAX) {
        uint old = atomic_inc(&nodeCounter[parent]);
        if (old == 0) {
            break;
        }

        uint lc = bvhNodes[parent].leftChildIndex;
        uint rc = bvhNodes[parent].rightChildIndex;
        aabb_union(&bvhNodes[lc].aabb, &bvhNodes[rc].aabb, &bvhNodes[parent].aabb);
        parent = parentIndices[parent];
    }
}
