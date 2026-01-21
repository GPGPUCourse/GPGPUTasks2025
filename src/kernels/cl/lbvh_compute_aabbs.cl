#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

static inline AABBGPU merge_aabb(AABBGPU a, AABBGPU b) {
    AABBGPU result;
    result.min_x = fmin(a.min_x, b.min_x);
    result.min_y = fmin(a.min_y, b.min_y);
    result.min_z = fmin(a.min_z, b.min_z);
    result.max_x = fmax(a.max_x, b.max_x);
    result.max_y = fmax(a.max_y, b.max_y);
    result.max_z = fmax(a.max_z, b.max_z);
    return result;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void lbvh_compute_aabbs(
    __global BVHNodeGPU* nodes,
    __global const AABBGPU* triangle_aabbs,
    __global const uint* leaf_triangle_indices,
    __global volatile int* node_visit_counts,
    uint nfaces)
{
    uint i = get_global_id(0);
    if (i >= nfaces) return;

    int leaf_idx = (int)nfaces - 1 + i;
    nodes[leaf_idx].aabb = triangle_aabbs[leaf_triangle_indices[i]];

    int current = -1;
    for (int p = 0; p < (int)nfaces - 1; p++) {
        if (nodes[p].leftChildIndex == (uint)leaf_idx || nodes[p].rightChildIndex == (uint)leaf_idx) {
            current = p;
            break;
        }
    }

    while (current >= 0) {
        int visit_count = atomic_inc(&node_visit_counts[current]);

        if (visit_count == 0) {
            return;
        }

        BVHNodeGPU node = nodes[current];
        AABBGPU left_aabb = nodes[node.leftChildIndex].aabb;
        AABBGPU right_aabb = nodes[node.rightChildIndex].aabb;
        nodes[current].aabb = merge_aabb(left_aabb, right_aabb);

        if (current == 0) break;

        int parent_idx = -1;
        for (int p = 0; p < (int)nfaces - 1; p++) {
            if (nodes[p].leftChildIndex == (uint)current || nodes[p].rightChildIndex == (uint)current) {
                parent_idx = p;
                break;
            }
        }

        current = parent_idx;
        if (current < 0) break;
    }
}
