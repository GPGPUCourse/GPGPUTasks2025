#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/bvh_node_gpu_shared.h"

__kernel void lbvh_5_refit(
    __global BVHNodeGPU* bvh_nodes,
    __global const uint* parents,
    __global int* flags,
    int n_faces)
{
    int i = get_global_id(0);
    if (i >= n_faces) return;

    // start at leaf node
    int current_node = (n_faces - 1) + i;

    while (true) {
        uint parent = parents[current_node];
        
        if (parent == 0xFFFFFFFF) {
            break; // Reached root or invalid
        }

        int old_val = atomic_inc(&flags[parent]);
        
        if (old_val == 0) {
            return;
        }
        
        uint left_idx = bvh_nodes[parent].leftChildIndex;
        uint right_idx = bvh_nodes[parent].rightChildIndex;

        AABBGPU left_aabb = bvh_nodes[left_idx].aabb;
        AABBGPU right_aabb = bvh_nodes[right_idx].aabb;
        
        AABBGPU aabb;
        aabb.min_x = min(left_aabb.min_x, right_aabb.min_x);
        aabb.min_y = min(left_aabb.min_y, right_aabb.min_y);
        aabb.min_z = min(left_aabb.min_z, right_aabb.min_z);
        aabb.max_x = max(left_aabb.max_x, right_aabb.max_x);
        aabb.max_y = max(left_aabb.max_y, right_aabb.max_y);
        aabb.max_z = max(left_aabb.max_z, right_aabb.max_z);
        
        bvh_nodes[parent].aabb = aabb;

        current_node = parent;
    }
}


