#include "../defines.h"
#include "helpers/rassert.cl"

#include "centroids_helpers.cl"

__attribute((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
reduce_centroids(
    __global CentroidGPU* mins,
    __global CentroidGPU* maxs,
    uint numGroups)
{
    const uint local_index = get_local_id(0);
    const uint local_size = get_local_size(0);

    __local float3 localMins[GROUP_SIZE];
    __local float3 localMaxs[GROUP_SIZE];

    float3 myMin = (float3)(+FLT_MAX, +FLT_MAX, +FLT_MAX);
    float3 myMax = (float3)(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (uint i = local_index; i < numGroups; i += local_size) {
        float3 gmin = loadCentroid(mins, i);
        float3 gmax = loadCentroid(maxs, i);

        myMin = (float3)(fmin(myMin.x, gmin.x),
            fmin(myMin.y, gmin.y),
            fmin(myMin.z, gmin.z));
        myMax = (float3)(fmax(myMax.x, gmax.x),
            fmax(myMax.y, gmax.y),
            fmax(myMax.z, gmax.z));
    }

    localMins[local_index] = myMin;
    localMaxs[local_index] = myMax;

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
        putCentroid(mins, 0, localMins[0]);
        putCentroid(maxs, 0, localMaxs[0]);
    }
}
