#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

#define INVALID 0xffffffff

void process_tri(
    __global const uint*       faces,
    __global const float*      vertices,
    uint                       tri_id,
    __global BVHNodeGPU*       node
) {
    uint3  f  = loadFace(faces, tri_id);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    node->aabb.min_x = min(v0.x, min(v1.x, v2.x));
    node->aabb.min_y = min(v0.y, min(v1.y, v2.y));
    node->aabb.min_z = min(v0.z, min(v1.z, v2.z));

    node->aabb.max_x = max(v0.x, max(v1.x, v2.x));
    node->aabb.max_y = max(v0.y, max(v1.y, v2.y));
    node->aabb.max_z = max(v0.z, max(v1.z, v2.z));

    node->leftChildIndex = INVALID;
    node->rightChildIndex = INVALID;
}

__kernel void lbvh_aabb_generation(
    __global const uint*       morton_codes,
    __global uint*             face_indexes,
    __global const uint*       faces,
    __global const float*      vertices,
    __global BVHNodeGPU*       nodes,
    __global int*              parent,
    __global int*              counter,
    uint                       nfaces,
    int                        calc_leafs
   )
{
    int i = get_global_id(0);
    const int leafStart = (int)nfaces - 1; 
    
    if (calc_leafs) { 
        if (i >= leafStart && i < nfaces + leafStart) {
            int face_id = face_indexes[i - leafStart];
            process_tri(faces, vertices, face_id - leafStart, &nodes[i]);
            // printf("%d <-- %d\n", i, parent[i]);
            atomic_add(&counter[parent[i]], 1);
            face_indexes[i - leafStart] -= leafStart;
        }
    }
    else {
        if (i < leafStart && counter[i] >= 2) {
            BVHNodeGPU l = nodes[nodes[i].leftChildIndex];
            BVHNodeGPU r = nodes[nodes[i].rightChildIndex];

            nodes[i].aabb.min_x = min(l.aabb.min_x, r.aabb.min_x);
            nodes[i].aabb.min_y = min(l.aabb.min_y, r.aabb.min_y);
            nodes[i].aabb.min_z = min(l.aabb.min_z, r.aabb.min_z);

            nodes[i].aabb.max_x = max(l.aabb.max_x, r.aabb.max_x);
            nodes[i].aabb.max_y = max(l.aabb.max_y, r.aabb.max_y);
            nodes[i].aabb.max_z = max(l.aabb.max_z, r.aabb.max_z);

            if (parent[i] != -1)
                atomic_add(&counter[parent[i]], 1);
        }
    }
}


