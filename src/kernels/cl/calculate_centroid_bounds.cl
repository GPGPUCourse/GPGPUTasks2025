
#include "helpers/rassert.cl"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"
#include "../defines.h"

inline void atomic_minf(volatile __global float *p, float val)
{
    float cur;
    while (val < (cur = *p))
        val = atomic_xchg(p, max(cur, val));
}

inline void atomic_maxf(volatile __global float *p, float val)
{
    float cur;
    while (val > (cur = *p))
        val = atomic_xchg(p, max(cur, val));
}

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void calculate_centroid_bounds(
    __global const float*     vertices,
    __global const uint*      faces,
    __global float* sceneMin,
    __global float* sceneMax,
             int n
)
{
    int i = get_global_id(0);
    if (i >= n) {
        return;
    }

    uint3  f = loadFace(faces, i);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);

    float3 c;
    c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
    c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
    c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);
    
    atomic_minf(&sceneMin[0], c.x);
    atomic_minf(&sceneMin[1], c.y);
    atomic_minf(&sceneMin[2], c.z);
    atomic_maxf(&sceneMax[0], c.x);
    atomic_maxf(&sceneMax[1], c.y);
    atomic_maxf(&sceneMax[2], c.z);

}
