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


void process_tri(
    __global const uint*       faces,
    __global const float*      vertices,
    uint                       tri_id,
    __global BVHNodeGPU*                node
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
}

__kernel void lbvh_aabb_generation(
    __global const uint*       morton_codes,
    __global uint*             face_indexes,
    __global const uint*       faces,
    __global const float*      vertices,
    __global BVHNodeGPU*       nodes,
    __global int*              parent,
    __global uint*             counter,
    __global uint*             terminated,
    uint                       nfaces
   )
{
    int i = get_global_id(0);
    const int leafStart = (int)nfaces - 1; 

    // calc aabb
    if (i < nfaces) {
        int face_id = face_indexes[i];
        if (face_id < leafStart || face_id >= nfaces + nfaces - 1)
            printf("EPTA1\n");
        process_tri(faces, vertices, face_id - leafStart, &nodes[i + leafStart]);

        int node_id = parent[i + leafStart];
        while (node_id != -1) {
            
            if (node_id < 0 || node_id >= nfaces + nfaces - 1) {
                printf("EPTA2.5 %d\n", node_id);
            }

            if (terminated[i] < 1) {
                atomic_add(&counter[node_id], 1);

                // printf("node_id = %d < nodes = %d\n", node_id, nfaces - 1);

                if (counter[node_id] == 1) {
                    atomic_add(&terminated[i], 1);
                }

                if (terminated[i] < 1) {
                    // printf("node = %d val = %d nfaces = %d\n", node_id, nodes[node_id].leftChildIndex, nfaces);
                    // printf("node = %d val = %d\n", node_id, nodes[node_id].rightChildIndex);

                    BVHNodeGPU l = nodes[nodes[node_id].leftChildIndex];
                    BVHNodeGPU r = nodes[nodes[node_id].rightChildIndex];
                    
                    if (counter[node_id] > 1) {
                        nodes[node_id].aabb.min_x = min(l.aabb.min_x, r.aabb.min_x);
                        nodes[node_id].aabb.min_y = min(l.aabb.min_y, r.aabb.min_y);
                        nodes[node_id].aabb.min_z = min(l.aabb.min_z, r.aabb.min_z);

                        nodes[node_id].aabb.max_x = max(l.aabb.max_x, r.aabb.max_x);
                        nodes[node_id].aabb.max_y = max(l.aabb.max_y, r.aabb.max_y);
                        nodes[node_id].aabb.max_z = max(l.aabb.max_z, r.aabb.max_z);
                    }
                }
            }

            barrier(CLK_GLOBAL_MEM_FENCE);
            
            node_id = parent[node_id];
        }
    }

    if (i == 0)
        printf("aabb build done\n");

    if (i < nfaces) {
        rassert(face_indexes >= leafStart, 90890356);
        face_indexes[i] -= leafStart;
    }

    if (i < leafStart) { 
        printf("cnt : %d, node = %d, l = %d, r = %d, has aabb = (%f,%f,%f) -- (%f,%f,%f)\n", counter[i], i,
        nodes[i].leftChildIndex, nodes[i].rightChildIndex,
        nodes[i].aabb.min_x, nodes[i].aabb.min_y, nodes[i].aabb.min_z,
        nodes[i].aabb.max_x, nodes[i].aabb.max_y, nodes[i].aabb.max_z);
    }
}


