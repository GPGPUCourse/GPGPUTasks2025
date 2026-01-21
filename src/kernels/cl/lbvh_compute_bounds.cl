#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void lbvh_compute_bounds(
    __global const float* triangle_centroids,
    __global float* bounds_min,
    __global float* bounds_max,
    uint nfaces)
{
    __local float local_min[3 * 256];
    __local float local_max[3 * 256];

    uint local_id = get_local_id(0);
    uint global_id = get_global_id(0);
    uint group_id = get_group_id(0);

    float3 local_min_val = (float3)(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 local_max_val = (float3)(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (uint i = global_id; i < nfaces; i += get_global_size(0)) {
        float3 c = (float3)(triangle_centroids[3*i], triangle_centroids[3*i+1], triangle_centroids[3*i+2]);
        local_min_val = fmin(local_min_val, c);
        local_max_val = fmax(local_max_val, c);
    }

    local_min[local_id * 3] = local_min_val.x;
    local_min[local_id * 3 + 1] = local_min_val.y;
    local_min[local_id * 3 + 2] = local_min_val.z;
    local_max[local_id * 3] = local_max_val.x;
    local_max[local_id * 3 + 1] = local_max_val.y;
    local_max[local_id * 3 + 2] = local_max_val.z;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            for (int axis = 0; axis < 3; axis++) {
                local_min[local_id * 3 + axis] = fmin(local_min[local_id * 3 + axis], local_min[(local_id + stride) * 3 + axis]);
                local_max[local_id * 3 + axis] = fmax(local_max[local_id * 3 + axis], local_max[(local_id + stride) * 3 + axis]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        for (int axis = 0; axis < 3; axis++) {
            atomic_min((volatile __global int*)&bounds_min[axis], as_int(local_min[axis]));
            atomic_max((volatile __global int*)&bounds_max[axis], as_int(local_max[axis]));
        }
    }
}
