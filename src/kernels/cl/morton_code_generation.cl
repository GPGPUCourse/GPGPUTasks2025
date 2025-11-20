#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"
#include "../shared_structs/morton_code_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
unsigned int expandBits(unsigned int v)
{
    // Ensure we have only lowest 10 bits
    // rassert(v == (v & 0x3FFu), 76389413321, v);

    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
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


// BVH traversal: closest hit along ray
__kernel void morton_code_generation(    
    uint                      nfaces,
    __global const float*     vertices,
    __global const uint*      faces,
    __global uint*            morton_encoding)
{
    uint i = get_global_id(0);

    if (i < nfaces) {
        uint3  f  = loadFace(faces, i);
        float3 v0 = loadVertex(vertices, f.x);
        float3 v1 = loadVertex(vertices, f.y);
        float3 v2 = loadVertex(vertices, f.z);

        float x = (v0.x + v1.x + v2.x) / 3;
        float y = (v0.y + v1.y + v2.y) / 3;
        float z = (v0.z + v1.z + v2.z) / 3;
        
        MortonCode baricenter = morton3D(x, y, z);

        morton_encoding[i] = baricenter;
    }
}
