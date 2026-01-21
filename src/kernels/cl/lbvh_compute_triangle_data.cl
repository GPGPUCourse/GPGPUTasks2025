#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"
#include "../shared_structs/aabb_gpu_shared.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void lbvh_compute_triangle_data(
    __global const float* vertices,
    __global const uint* faces,
    __global AABBGPU* triangle_aabbs,
    __global float* triangle_centroids,
    uint nfaces)
{
    uint i = get_global_id(0);
    if (i >= nfaces) return;

    uint3 f = (uint3)(faces[3*i], faces[3*i+1], faces[3*i+2]);
    float3 v0 = (float3)(vertices[3*f.x], vertices[3*f.x+1], vertices[3*f.x+2]);
    float3 v1 = (float3)(vertices[3*f.y], vertices[3*f.y+1], vertices[3*f.y+2]);
    float3 v2 = (float3)(vertices[3*f.z], vertices[3*f.z+1], vertices[3*f.z+2]);

    AABBGPU aabb;
    aabb.min_x = fmin(fmin(v0.x, v1.x), v2.x);
    aabb.min_y = fmin(fmin(v0.y, v1.y), v2.y);
    aabb.min_z = fmin(fmin(v0.z, v1.z), v2.z);
    aabb.max_x = fmax(fmax(v0.x, v1.x), v2.x);
    aabb.max_y = fmax(fmax(v0.y, v1.y), v2.y);
    aabb.max_z = fmax(fmax(v0.z, v1.z), v2.z);

    triangle_aabbs[i] = aabb;

    float3 centroid = (v0 + v1 + v2) / 3.0f;
    triangle_centroids[3*i] = centroid.x;
    triangle_centroids[3*i+1] = centroid.y;
    triangle_centroids[3*i+2] = centroid.z;
}
