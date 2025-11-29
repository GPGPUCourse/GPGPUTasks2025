#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/bvh_node_gpu_shared.h"
#include "geometry_helpers.cl"

static inline AABBGPU merge_aabb(AABBGPU left, AABBGPU right) {
    AABBGPU result;
    result.min_x = min(left.min_x, right.min_x);
    result.min_y = min(left.min_y, right.min_y);
    result.min_z = min(left.min_z, right.min_z);
    result.max_x = max(left.max_x, right.max_x);
    result.max_y = max(left.max_y, right.max_y);
    result.max_z = max(left.max_z, right.max_z);
    return result;
}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void bottom_up_aabb(
    __global BVHNodeGPU* nodes,
    __global int* pending,
    const uint nFaces
) {
    const int i = get_global_id(0);
    if (i >= nFaces) return;
    const int leafStart = nFaces - 1;

    int currentNode = leafStart + i; // Start as leaf node

    while (currentNode > 0) {
        const uint parent = nodes[currentNode].parentIndex;
        const int afterDec = atomic_dec(&pending[parent]);
        if (afterDec == 1) {
            const uint leftChild = nodes[parent].leftChildIndex;
            const uint rightChild = nodes[parent].rightChildIndex;
            nodes[parent].aabb = merge_aabb(nodes[leftChild].aabb, nodes[rightChild].aabb);
            currentNode = parent;
        } else {
            // we die :[
            // but others will live!!!
            break;
        }
    }
}