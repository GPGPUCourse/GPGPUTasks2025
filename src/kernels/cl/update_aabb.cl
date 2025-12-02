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

static inline AABBGPU merge_aabb(AABBGPU first, AABBGPU second)
{
    AABBGPU aabb;
    aabb.min_x = min(first.min_x, second.min_x);
    aabb.min_y = min(first.min_y, second.min_y);
    aabb.min_z = min(first.min_z, second.min_z);
    aabb.max_x = max(first.max_x, second.max_x);
    aabb.max_y = max(first.max_y, second.max_y);
    aabb.max_z = max(first.max_z, second.max_z);
    return aabb;
}

__kernel void update_aabb(
    __global const uint* parents,
    __global       BVHNodeGPU* output_nodes,
                   int parent_size)
{
    const unsigned int i = get_global_id(0);
    if (i >= parent_size) {
        return;
    }
    int index = i;
    while (index != 0) {
        // At least on the second pass aabb should be correct
        BVHNodeGPU node = output_nodes[parents[index]];
        AABBGPU tmp = merge_aabb(output_nodes[node.leftChildIndex].aabb, output_nodes[node.rightChildIndex].aabb);
        output_nodes[parents[index]].aabb = merge_aabb(node.aabb, tmp);
        index = parents[index];
    }
}
