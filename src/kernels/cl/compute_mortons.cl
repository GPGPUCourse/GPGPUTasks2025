#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "helpers/rassert.cl"
#include "random_helpers.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
compute_mortons(
    __global const uint *faces,
    __global const float *vertices,
    uint nfaces,
    const float cMinX,
    const float cMinY,
    const float cMinZ,
    const float cMaxX,
    const float cMaxY,
    const float cMaxZ,
    __global uint *morton_codes)
{
    const uint index = get_global_id(0);
    if (index >= nfaces)
    {
        return;
    }

    float3 cMin, cMax;
    cMin.x = cMinX;
    cMin.y = cMinY;
    cMin.z = cMinZ;
    cMax.x = cMaxX;
    cMax.y = cMaxY;
    cMax.z = cMaxZ;
    const float eps = 1e-9f;
    const float dx = max(cMax.x - cMin.x, eps);
    const float dy = max(cMax.y - cMin.y, eps);
    const float dz = max(cMax.z - cMin.z, eps);

    uint3 f = loadFace(faces, index);
    // printf("Face %u: (%u, %u, %u), with limit %u\n", index, f.x, f.y, f.z, nfaces);
    // rassert(f.x < nfaces && f.y < nfaces && f.z < nfaces, 123456);
    float3 v0 = loadVertex(vertices, f.x);
    float3 v1 = loadVertex(vertices, f.y);
    float3 v2 = loadVertex(vertices, f.z);
    /*
    printf("Vertices: v0(%f, %f, %f), v1(%f, %f, %f), v2(%f, %f, %f)\n",
           v0.x, v0.y, v0.z,
           v1.x, v1.y, v1.z,
           v2.x, v2.y, v2.z);
           */

    float3 c;
    c.x = (v0.x + v1.x + v2.x) / 3.0f;
    c.y = (v0.y + v1.y + v2.y) / 3.0f;
    c.z = (v0.z + v1.z + v2.z) / 3.0f;
    float nx = (c.x - cMin.x) / dx;
    float ny = (c.y - cMin.y) / dy;
    float nz = (c.z - cMin.z) / dz;

    nx = min(max(nx, 0.0f), 1.0f);
    ny = min(max(ny, 0.0f), 1.0f);
    nz = min(max(nz, 0.0f), 1.0f);
    // printf("Centroid for face %u: (%f, %f, %f)\n", index, c.x, c.y, c.z);
    // printf("Normalized centroid for face %u: (%f, %f, %f)\n", index, nx, ny, nz);

    morton_codes[index] = getMortonCode(nx, ny, nz);
    // printf("Morton code for face %u: %u\n", index, morton_codes[index]);
}