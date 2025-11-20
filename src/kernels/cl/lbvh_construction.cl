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

inline int common_pref_len(
    __global const uint*       morton_codes,
    int l,
    int r, 
    int nfaces
) 
{
    if (l > r) {
        int tmp = r;
        r = l;
        l = tmp;
    }

    rassert(r == l + 1, 4358904);

    if (l < 0)
        return -1;

    if (r >= nfaces)
        return -1;

    int bit = 31;
    while (bit >= 0 && (((morton_codes[l] + l) >> bit) & 1) == (((morton_codes[r] + r) >> bit) & 1))
        bit--;
    
    return 32 - bit - 1;
}

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

__kernel void lbvh_construction(
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

    // printf("%d has morton code %d\n", i, morton_codes[i]);

    // if (i == 0) {
    //     printf("test = %d\n", common_pref_len(morton_codes, 0, 1));
    //     printf("test = %d\n", common_pref_len(morton_codes, 3, 4));
    //     printf("test = %d\n", common_pref_len(morton_codes, 0, 1));
    // }

    int d, pref_len, j, min_pref, min_pref_ind;
    if (i < leafStart) {
        d = 0, pref_len = -1;

        if (i == 0)
            d = +1;
        else
            d = (common_pref_len(morton_codes, i, i + 1, nfaces) > common_pref_len(morton_codes, i - 1, i, nfaces) ? +1 : -1);

        pref_len = common_pref_len(morton_codes, i, i - d, nfaces);

        j = i;
        min_pref = 100, min_pref_ind = -1;
        while (j + d < nfaces &&
               j + d >= 0     &&
               pref_len < common_pref_len(morton_codes, j, j + d, nfaces)) { 
            if (min_pref > common_pref_len(morton_codes, j, j + d, nfaces)) {
                min_pref = common_pref_len(morton_codes, j, j + d, nfaces);
                min_pref_ind = j;
            }
            j += d;
        }
        // min_pref_ind, min_pref_ind + d - max on segment

        if (i == 0) {
            parent[0] = -1;
        }
        if (j == i + d) {
            nodes[i].leftChildIndex  = face_indexes[min(i, j)];
            nodes[i].rightChildIndex = face_indexes[max(i, j)];
        }
        else if (i == min_pref_ind) {
            if (d == +1) {
                nodes[i].leftChildIndex  = face_indexes[i];
                nodes[i].rightChildIndex = i + d;
            }
            if (d == -1) {
                nodes[i].leftChildIndex  = i + d;
                nodes[i].rightChildIndex = face_indexes[i];
            }
        }
        else if (min_pref_ind + d == j) { 
            if (d == +1) {
                nodes[i].leftChildIndex  = j - d;
                nodes[i].rightChildIndex = face_indexes[j];
            }
            if (d == -1) {
                nodes[i].leftChildIndex  = face_indexes[j];
                nodes[i].rightChildIndex = j - d;
            }
        }
        else {
            nodes[i].leftChildIndex  = min(min_pref_ind, min_pref_ind + d);
            nodes[i].rightChildIndex = max(min_pref_ind, min_pref_ind + d);
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (i < leafStart) { 
        parent[nodes[i].leftChildIndex] = i;
        parent[nodes[i].rightChildIndex] = i;
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (i == 0)
        printf("tree build done\n");


    if (i < nfaces) {
        int node_id = i + leafStart;

        while (node_id != -1) {
            if (node_id < 0 || node_id > nfaces + nfaces - 1)
                printf("%d on i = %d\n", node_id, i);
            node_id = parent[node_id];
        }
    }

    // calc aabb
    if (i < nfaces) {
        int face_id = face_indexes[i];
        if (face_id < leafStart || face_id >= nfaces + nfaces - 1)
            printf("EPTA1\n");
        process_tri(faces, vertices, face_id - leafStart, &nodes[i + leafStart]);

        int node_id = i + leafStart;
        while (true) {
            if (node_id < 0 || node_id >= nfaces + nfaces - 1) {
                printf("EPTA2.5 %d\n", node_id);
            }

            node_id = parent[node_id];

            if (node_id == -1)
                break;

            if (terminated[i] < 1) {
                atomic_add(&counter[node_id], 1);

                // printf("node_id = %d < nodes = %d\n", node_id, nfaces - 1);

                if (counter[node_id] == 1) {
                    atomic_add(&terminated[i], 1);
                    continue;
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
        }
    }

    if (i == 0)
        printf("aabb build done\n");

    if (i < nfaces) {
        rassert(face_indexes >= leafStart, 90890356);
        face_indexes[i] -= leafStart;
    }

    // if (i < leafStart) { 
    //     printf("node = %d, l = %d, r = %d, has aabb = (%f,%f,%f) -- (%f,%f,%f)\n", i,
    //     nodes[i].leftChildIndex, nodes[i].rightChildIndex,
    //     nodes[i].aabb.min_x, nodes[i].aabb.min_y, nodes[i].aabb.min_z,
    //     nodes[i].aabb.max_x, nodes[i].aabb.max_y, nodes[i].aabb.max_z);
    // }
}
