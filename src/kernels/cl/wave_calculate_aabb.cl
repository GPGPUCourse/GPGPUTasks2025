
#include "helpers/rassert.cl"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

#include "../defines.h"

static inline AABBGPU aabb_from_leaf(
    __global const int* leaf_indices,
    __global const float*     vertices,
    __global const uint*      faces,
             int leaf_index,
             int leaf_start,
             int n
)
{
    AABBGPU res;
    uint triIndex = leaf_indices[leaf_index - leaf_start];
    uint3 face = loadFace(faces, triIndex);
    float3 v;
    // printf("li=%d, -> %d %d %d ;; %d\n", leaf_index, face.x, face.y, face.z, triIndex);
    v = loadVertex(vertices, face.x);
    res.min_x = v.x;
    res.max_x = v.x;
    res.min_y = v.y;
    res.max_y = v.y;
    res.min_z = v.z;
    res.max_z = v.z;
    // printf("li=%d, -1- %d %d %d\n", leaf_index, v.x, v.y, v.z);

    v = loadVertex(vertices, face.y);
    res.min_x = min(res.min_x, v.x);
    res.max_x = max(res.max_x, v.x);
    res.min_y = min(res.min_y, v.y);
    res.max_y = max(res.max_y, v.y);
    res.min_z = min(res.min_z, v.z);
    res.max_z = max(res.max_z, v.z);
    // printf("li=%d, -2- %d %d %d\n", leaf_index, v.x, v.y, v.z);

    v = loadVertex(vertices, face.z);
    res.min_x = min(res.min_x, v.x);
    res.max_x = max(res.max_x, v.x);
    res.min_y = min(res.min_y, v.y);
    res.max_y = max(res.max_y, v.y);
    res.min_z = min(res.min_z, v.z);
    res.max_z = max(res.max_z, v.z);
    // printf("li=%d, -3- %d %d %d\n", leaf_index, v.x, v.y, v.z);

    return res;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void wave_calculate_aabb(
    __global BVHNodeGPU* lbvh,
    volatile __global int* ready_flags,
    __global const int* leaf_indices,
    __global const float*     vertices,
    __global const uint*      faces,
             int n
)
{
    int i = get_global_id(0);
    const int leafStart = (int)n - 1;
    if (i >= n-1 + n) {
        return;
    }
    if (ready_flags[i]) {
        return;
    }
    if (i >= leafStart) {
        lbvh[i].aabb = aabb_from_leaf(leaf_indices, vertices, faces, i, leafStart, n);
        lbvh[i].leftChildIndex = 2*n;
        lbvh[i].rightChildIndex = 2*n;
        atomic_add(&ready_flags[i], 1);
        return;
    }
    AABBGPU left, right;
    int left_ind = lbvh[i].leftChildIndex;
    if (!atomic_add(&ready_flags[left_ind], 0)) {
        return;
    }
    left = lbvh[left_ind].aabb;

    int right_ind = lbvh[i].rightChildIndex;
    if (!atomic_add(&ready_flags[right_ind], 0)) {
        return;
    }
    right = lbvh[right_ind].aabb;
    AABBGPU res;
    res.min_x = min(left.min_x, right.min_x);
    res.max_x = max(left.max_x, right.max_x);
    res.min_y = min(left.min_y, right.min_y);
    res.max_y = max(left.max_y, right.max_y);
    res.min_z = min(left.min_z, right.min_z);
    res.max_z = max(left.max_z, right.max_z);
    lbvh[i].aabb = res;
    atomic_add(&ready_flags[i], 1);
    // printf("gotcha i=%d, left=%d, right=%d, [%g; %g] x [%g; %g] x [%g; %g]\n[%g; %g] x [%g; %g] x [%g; %g]\n[%g; %g] x [%g; %g] x [%g; %g]\n", i, left_ind, right_ind, res.min_x, res.max_x, res.min_y, res.max_y, res.min_z, res.max_z, left.min_x, left.max_x, left.min_y, left.max_y, left.min_z, left.max_z, right.min_x, right.max_x, right.min_y, right.max_y, right.min_z, right.max_z);
}
