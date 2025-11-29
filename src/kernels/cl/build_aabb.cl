#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "helpers/rassert.cl"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
build_aabb(__global const uint *used,
           __global uint *now_used,
           __global uint *changed,
           uint nfaces,
           __global BVHNodeGPU *lbvh)
{
    const uint index = get_global_id(0);
    if (index >= nfaces)
    {
        return;
    }
    now_used[index] = used[index];
    if (used[index])
    {
        return;
    }
    uint left = lbvh[index].leftChildIndex;
    uint right = lbvh[index].rightChildIndex;
    if (left < nfaces && right < nfaces)
    {
        if (!used[left] || !used[right])
        {
            return;
        }
    }
    atomic_add(changed, 1);
    rassert(left < nfaces, 4311);
    rassert(right < nfaces, 4313);
    rassert(index < nfaces, 4312);
    now_used[index] = 1;
    AABBGPU left_aabb = lbvh[left].aabb;
    AABBGPU right_aabb = lbvh[right].aabb;
    lbvh[index].aabb.min_x = fmin(left_aabb.min_x, right_aabb.min_x);
    lbvh[index].aabb.min_y = fmin(left_aabb.min_y, right_aabb.min_y);
    lbvh[index].aabb.min_z = fmin(left_aabb.min_z, right_aabb.min_z);
    lbvh[index].aabb.max_x = fmax(left_aabb.max_x, right_aabb.max_x);
    lbvh[index].aabb.max_y = fmax(left_aabb.max_y, right_aabb.max_y);
    lbvh[index].aabb.max_z = fmax(left_aabb.max_z, right_aabb.max_z);
}