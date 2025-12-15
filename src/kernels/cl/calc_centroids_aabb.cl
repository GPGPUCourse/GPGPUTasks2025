#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

inline float3 calcCentroid(const float3 v0, const float3 v1, const float3 v2)
{
    return (float3)((v0.x + v1.x + v2.x) * (1.0f / 3.0f), (v0.y + v1.y + v2.y) * (1.0f / 3.0f), (v0.z + v1.z + v2.z) * (1.0f / 3.0f));
}

__kernel void calc_centroids_aabb(
    __global const float* vertices,
    __global const uint* faces,
    const uint nfaces,
    __global uint* triIndexes,
    __global float* centroidsX, __global float* centroidsY, __global float* centroidsZ,
    __global float* aabbXMin, __global float* aabbXMax,
    __global float* aabbYMin, __global float* aabbYMax,
    __global float* aabbZMin, __global float* aabbZMax)
{
    const uint i = get_global_id(0);
    const uint localI = get_local_id(0);
    const uint groupI = get_group_id(0);
    if (i < nfaces) {
        const uint3 face = loadFace(faces, i);
        const float3 v0 = loadVertex(vertices, face.x);
        const float3 v1 = loadVertex(vertices, face.y);
        const float3 v2 = loadVertex(vertices, face.z);
        const float3 centroid = calcCentroid(v0, v1, v2);

        triIndexes[i] = i;

        centroidsX[i] = centroid.x;
        centroidsY[i] = centroid.y;
        centroidsZ[i] = centroid.z;

        aabbXMin[i] = fmin(v0.x, fmin(v1.x, v2.x));
        aabbXMax[i] = fmax(v0.x, fmax(v1.x, v2.x));
        aabbYMin[i] = fmin(v0.y, fmin(v1.y, v2.y));
        aabbYMax[i] = fmax(v0.y, fmax(v1.y, v2.y));
        aabbZMin[i] = fmin(v0.z, fmin(v1.z, v2.z));
        aabbZMax[i] = fmax(v0.z, fmax(v1.z, v2.z));
    }
}