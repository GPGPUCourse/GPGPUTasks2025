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
build_aabb_leaves(__global const float *vertices,
                  __global const uint *faces,
                  __global const uint *indices,
                  uint nfaces,
                  __global BVHNodeGPU *lbvh)
{
    const uint index = get_global_id(0);
    if (index >= nfaces)
    {
        return;
    }

    uint3 f = loadFace(faces, indices[index]);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    uint leafIndex = index + nfaces - 1;
    lbvh[leafIndex].aabb.min_x = fmin(fmin(v0.x, v1.x), v2.x);
    lbvh[leafIndex].aabb.min_y = fmin(fmin(v0.y, v1.y), v2.y);
    lbvh[leafIndex].aabb.min_z = fmin(fmin(v0.z, v1.z), v2.z);
    lbvh[leafIndex].aabb.max_x = fmax(fmax(v0.x, v1.x), v2.x);
    lbvh[leafIndex].aabb.max_y = fmax(fmax(v0.y, v1.y), v2.y);
    lbvh[leafIndex].aabb.max_z = fmax(fmax(v0.z, v1.z), v2.z);
    lbvh[leafIndex].leftChildIndex = 0xFFFFFFFF;
    lbvh[leafIndex].rightChildIndex = 0xFFFFFFFF;
}