#include "../shared_structs/centroid_gpu_shared.h"

static inline void putCentroid(__global CentroidGPU* centroids, uint index, float3 data)
{
    centroids[index].x = data.x;
    centroids[index].y = data.y;
    centroids[index].z = data.z;
}

static inline float3 loadCentroid(__global const CentroidGPU* centroids, uint index)
{
    return (float3)(centroids[index].x, centroids[index].y, centroids[index].z);
}
