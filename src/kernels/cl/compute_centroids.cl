#include "../defines.h"
#include "helpers/rassert.cl"

#include "centroids_helpers.cl"
#include "geometry_helpers.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
compute_centroids(
    __global const float* vertices,
    __global const uint* faces,
    __global CentroidGPU* centroids, // [nfaces]
    __global CentroidGPU* groupMins, // [numGroups]
    __global CentroidGPU* groupMaxs, // [numGroups]
    uint nfaces)
{
    const uint global_index = get_global_id(0);
    const uint local_index = get_local_id(0);
    const uint group_index = get_group_id(0);
    const uint local_size = get_local_size(0);

    __local float3 localMins[GROUP_SIZE];
    __local float3 localMaxs[GROUP_SIZE];

    float3 c;

    if (global_index < nfaces) {
        uint3 face = loadFace(faces, global_index);
        float3 v0 = loadVertex(vertices, face.x);
        float3 v1 = loadVertex(vertices, face.y);
        float3 v2 = loadVertex(vertices, face.z);

        const float inv3 = 1.0f / 3.0f;
        c.x = (v0.x + v1.x + v2.x) * inv3;
        c.y = (v0.y + v1.y + v2.y) * inv3;
        c.z = (v0.z + v1.z + v2.z) * inv3;

        putCentroid(centroids, global_index, c);

        localMins[local_index] = c;
        localMaxs[local_index] = c;
    } else {
        localMins[local_index] = (float3)(+FLT_MAX, +FLT_MAX, +FLT_MAX);
        localMaxs[local_index] = (float3)(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint offset = local_size >> 1; offset > 0; offset >>= 1) {
        if (local_index < offset) {
            float3 aMin = localMins[local_index];
            float3 bMin = localMins[local_index + offset];
            float3 aMax = localMaxs[local_index];
            float3 bMax = localMaxs[local_index + offset];

            localMins[local_index] = (float3)(fmin(aMin.x, bMin.x),
                fmin(aMin.y, bMin.y),
                fmin(aMin.z, bMin.z));

            localMaxs[local_index] = (float3)(fmax(aMax.x, bMax.x),
                fmax(aMax.y, bMax.y),
                fmax(aMax.z, bMax.z));
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        putCentroid(groupMins, group_index, localMins[0]);
        putCentroid(groupMaxs, group_index, localMaxs[0]);
    }
}
