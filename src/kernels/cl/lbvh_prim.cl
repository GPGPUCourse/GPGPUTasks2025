#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

unsigned expandBits(unsigned v)
{
    // Ensure we have only lowest 10 bits
    rassert(v == (v & 0x3FFu), 76389413321);

    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

MortonCode morton3D(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    unsigned int ix = min(max((int) (x * 1024.0f), 0), 1023);
    unsigned int iy = min(max((int) (y * 1024.0f), 0), 1023);
    unsigned int iz = min(max((int) (z * 1024.0f), 0), 1023);

    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

// Build LBVH (Karras 2013) on CPU.
// Output:
//   outNodes           - BVH nodes array of size (2*N - 1). Root is node 0.
//   outLeafTriIndices  - size N, mapping leaf i -> original triangle index.
//
// Node indexing convention (matches LBVH style):
//   N = number of triangles (faces.size()).
//   Internal nodes: indices [0 .. N-2]
//   Leaf nodes:     indices [N-1 .. 2*N-2]
//   Leaf at index (N-1 + i) corresponds to outLeafTriIndices[i].
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void lbvh_prim(
    unsigned N,
    __global const unsigned* faces,
    __global const float* vertices,
    __global BVHThinNodeGPU* outNodes,
    __global BVHTriangleGPU* outTris,
    __global uint* mortons,
    float cMinX, float cMinY, float cMinZ,
    float cMaxX, float cMaxY, float cMaxZ,
    __global uint gHist[4][256]
)
{
    unsigned i = get_global_id(0);
    unsigned li = get_local_id(0);
    
    __local uint hist[4][256];
    if(li < 256) {
        for(unsigned j = 0; j < 4; j++)
            hist[j][li] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(i < N) {
        //printf("%d\n", i);
        unsigned f0 = faces[3 * i], f1 = faces[3 * i + 1], f2 = faces[3 * i + 2];
        float3 v0 = {vertices[3 * f0], vertices[3 * f0 + 1], vertices[3 * f0 + 2]};
        float3 v1 = {vertices[3 * f1], vertices[3 * f1 + 1], vertices[3 * f1 + 2]};
        float3 v2 = {vertices[3 * f2], vertices[3 * f2 + 1], vertices[3 * f2 + 2]};

        // Triangle AABB
        AABBGPU aabb;
        aabb.min_x = min(v0.x, min(v1.x, v2.x));
        aabb.min_y = min(v0.y, min(v1.y, v2.y));
        aabb.min_z = min(v0.z, min(v1.z, v2.z));
        aabb.max_x = max(v0.x, max(v1.x, v2.x));
        aabb.max_y = max(v0.y, max(v1.y, v2.y));
        aabb.max_z = max(v0.z, max(v1.z, v2.z));

        // // Centroid
        // float3 c;
        // c.x = (v0.x + v1.x + v2.x) * (1.0f / 3.0f);
        // c.y = (v0.y + v1.y + v2.y) * (1.0f / 3.0f);
        // c.z = (v0.z + v1.z + v2.z) * (1.0f / 3.0f);
        // AABB centroid (as described in papers)
        float3 c;
        c.x = (aabb.min_x + aabb.max_x) * 0.5f;
        c.y = (aabb.min_y + aabb.max_y) * 0.5f;
        c.z = (aabb.min_z + aabb.max_z) * 0.5f;

        const float eps = 1e-9f;
        const float dx = max(cMaxX - cMinX, eps);
        const float dy = max(cMaxY - cMinY, eps);
        const float dz = max(cMaxZ - cMinZ, eps);

        float nx = (c.x - cMinX) / dx;
        float ny = (c.y - cMinY) / dy;
        float nz = (c.z - cMinZ) / dz;

        // Clamp to [0,1]
        nx = min(max(nx, 0.0f), 1.0f);
        ny = min(max(ny, 0.0f), 1.0f);
        nz = min(max(nz, 0.0f), 1.0f);

        unsigned morton = morton3D(nx, ny, nz);
        mortons[i] = morton;
        ((ulong)morton) << 32 | i;
        outNodes[i].aabb = aabb;
        outNodes[i].children[0] = i;
        outNodes[i].children[1] = ~(i * sizeof(BVHTriangleGPU) >> 3);
        outTris[i].a[0] = v0.x;
        outTris[i].a[1] = v0.y;
        outTris[i].a[2] = v0.z;
        outTris[i].b[0] = v1.x;
        outTris[i].b[1] = v1.y;
        outTris[i].b[2] = v1.z;
        outTris[i].c[0] = v2.x;
        outTris[i].c[1] = v2.y;
        outTris[i].c[2] = v2.z;
        outTris[i].c[2] = v2.z;
        outTris[i].triangle_id = i;

        unsigned digit0 = morton & 255;
        unsigned digit1 = morton >> 8 & 255;
        unsigned digit2 = morton >> 16 & 255;
        unsigned digit3 = morton >> 24 & 255;
        
        atomic_inc(&hist[0][digit0]);
        atomic_inc(&hist[1][digit1]);
        atomic_inc(&hist[2][digit2]);
        atomic_inc(&hist[3][digit3]);
        
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    if(li < 256) {
        for(unsigned j = 0; j < 4; j++)
        {
            atomic_add(&gHist[j][li], hist[j][li]);
        }
    }
}
