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

static inline AABBGPU compute_leaf_aabb(
    __global const float*     vertices,
    __global const uint*      faces,
    uint                      faceId)
{
    AABBGPU aabb;

    uint3 f = loadFace(faces, faceId);
    float3 a = loadVertex(vertices, f.x);
    float3 b = loadVertex(vertices, f.y);
    float3 c = loadVertex(vertices, f.z);

    aabb.min_x = min(a.x, min(b.x, c.x));
    aabb.min_y = min(a.y, min(b.y, c.y));
    aabb.min_z = min(a.z, min(b.z, c.z));
    aabb.max_x = max(a.x, max(b.x, c.x));
    aabb.max_y = max(a.y, max(b.y, c.y));
    aabb.max_z = max(a.z, max(b.z, c.z));
    return aabb;
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
up_merge_aabb(
    __global const float*      vertices,
    __global const uint*       faces,
    __global const uint*       parent,
    __global const uint*       leafTriIndices,
    __global BVHNodeGPU*       bvhNodes,
    __global int*              counters,
    const uint                 N)
{
    uint i = get_global_id(0);
    if (i >= N) {
        return;
    }

    AABBGPU cur = compute_leaf_aabb(vertices, faces, leafTriIndices[i]);
    i += N - 1;

    bvhNodes[i].aabb = cur;

    while (i != 0) {
        const uint j = parent[i];
        
        const int was = atomic_inc(&counters[j]);
        
        if (was == 0) {
            break;
        }

        const uint otherChildIndex = bvhNodes[j].leftChildIndex ^ bvhNodes[j].rightChildIndex ^ i;

        AABBGPU other = bvhNodes[otherChildIndex].aabb;
        cur.min_x = min(cur.min_x, other.min_x);
        cur.min_y = min(cur.min_y, other.min_y);
        cur.min_z = min(cur.min_z, other.min_z);
        cur.max_x = max(cur.max_x, other.max_x);
        cur.max_y = max(cur.max_y, other.max_y);
        cur.max_z = max(cur.max_z, other.max_z);

        bvhNodes[j].aabb = cur;

        i = j;
    }
}