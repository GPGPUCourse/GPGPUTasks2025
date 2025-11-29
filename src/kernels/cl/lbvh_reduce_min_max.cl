#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#include "../shared_structs/aabb_gpu_shared.h"

__kernel void lbvh_reduce_min_max(
    __global const AABBGPU*    input,
    __global AABBGPU*          output,
    uint                       n)
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

    for (uint i = globalIdx * BOX_BLOCK_SIZE; i < min((globalIdx + 1) * BOX_BLOCK_SIZE, n); ++i) {
        AABBGPU bbox = input[i];

        aabbMinMax.min_x = min(aabbMinMax.min_x, bbox.min_x);
        aabbMinMax.min_y = min(aabbMinMax.min_y, bbox.min_y);
        aabbMinMax.min_z = min(aabbMinMax.min_z, bbox.min_z);
        aabbMinMax.max_x = max(aabbMinMax.max_x, bbox.max_x);
        aabbMinMax.max_y = max(aabbMinMax.max_y, bbox.max_y);
        aabbMinMax.max_z = max(aabbMinMax.max_z, bbox.max_z);
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

        output[get_group_id(0)] = aabbMinMax;
    }
}
