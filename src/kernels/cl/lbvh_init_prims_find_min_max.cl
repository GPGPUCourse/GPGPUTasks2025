#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "geometry_helpers.cl"

__kernel void lbvh_init_prims_find_min_max(
    __global const float*      vertices,
    __global const uint*       faces,
    __global BVHPrimGPU*       prims,
    __global AABBGPU*          minmax,
    uint                       nfaces)
{
    const uint globalIdx = get_global_id(0);
    const uint localIdx = get_local_id(0);

    __local AABBGPU bboxes[GROUP_SIZE];

    AABBGPU aabbMinMax;
    aabbMinMax.min_x = +INFINITY;
    aabbMinMax.min_y = +INFINITY;
    aabbMinMax.min_z = +INFINITY;
    aabbMinMax.max_x = -INFINITY;
    aabbMinMax.max_y = -INFINITY;
    aabbMinMax.max_z = -INFINITY;

    for (uint i = globalIdx * BOX_BLOCK_SIZE; i < min((globalIdx + 1) * BOX_BLOCK_SIZE, nfaces); ++i) {
        BVHPrimGPU prim;
        uint3  f = loadFace(faces, i);
        float3 a = loadVertex(vertices, f.x);
        float3 b = loadVertex(vertices, f.y);
        float3 c = loadVertex(vertices, f.z);

        prim.aabb.min_x = min(a.x, min(b.x, c.x));
        prim.aabb.min_y = min(a.y, min(b.y, c.y));
        prim.aabb.min_z = min(a.z, min(b.z, c.z));
        prim.aabb.max_x = max(a.x, max(b.x, c.x));
        prim.aabb.max_y = max(a.y, max(b.y, c.y));
        prim.aabb.max_z = max(a.z, max(b.z, c.z));

        prim.centroidX = (a.x + b.x + c.x) / 3.0f;
        prim.centroidY = (a.y + b.y + c.y) / 3.0f;
        prim.centroidZ = (a.z + b.z + c.z) / 3.0f;

        prim.triIndex = i;

        prims[i] = prim;

        aabbMinMax.min_x = min(aabbMinMax.min_x, prim.centroidX);
        aabbMinMax.min_y = min(aabbMinMax.min_y, prim.centroidY);
        aabbMinMax.min_z = min(aabbMinMax.min_z, prim.centroidZ);
        aabbMinMax.max_x = max(aabbMinMax.max_x, prim.centroidX);
        aabbMinMax.max_y = max(aabbMinMax.max_y, prim.centroidY);
        aabbMinMax.max_z = max(aabbMinMax.max_z, prim.centroidZ);
    }

    bboxes[localIdx] = aabbMinMax;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localIdx == 0) {
        for (uint i = 1; i < GROUP_SIZE; ++i) {
            AABBGPU bbox = bboxes[i];
            
            aabbMinMax.min_x = min(aabbMinMax.min_x, bbox.min_x);
            aabbMinMax.min_y = min(aabbMinMax.min_y, bbox.min_y);
            aabbMinMax.min_z = min(aabbMinMax.min_z, bbox.min_z);
            aabbMinMax.max_x = max(aabbMinMax.max_x, bbox.max_x);
            aabbMinMax.max_y = max(aabbMinMax.max_y, bbox.max_y);
            aabbMinMax.max_z = max(aabbMinMax.max_z, bbox.max_z);
        }

        minmax[get_group_id(0)] = aabbMinMax;
    }
}
