#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

#include "../shared_structs/morton_code_gpu_shared.h"

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
inline uint expandBits(uint v)
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

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
inline MortonCode morton3D(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    uint ix = min(max((int)(x * 1024.0f), 0), 1023);
    uint iy = min(max((int)(y * 1024.0f), 0), 1023);
    uint iz = min(max((int)(z * 1024.0f), 0), 1023);

    uint xx = expandBits(ix);
    uint yy = expandBits(iy);
    uint zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

__kernel void calc_morton(
    __global const float *centroidsX, __global const float *centroidsY, __global const float *centroidsZ,
    __global const float *aabbXMin, __global const float *aabbXMax,
    __global const float *aabbYMin, __global const float *aabbYMax,
    __global const float *aabbZMin, __global const float *aabbZMax,
    const float cXMin, const float cXMax,
    const float cYMin, const float cYMax,
    const float cZMin, const float cZMax,
    const uint nfaces,
    __global MortonCode *codes)
{
    const uint i = get_global_id(0);
    if (i >= nfaces)
    {
        return;
    }
    const float eps = 1e-9f;
    const float dx = fmax(cXMax - cXMin, eps);
    const float dy = fmax(cYMax - cYMin, eps);
    const float dz = fmax(cZMax - cZMin, eps);

    const float3 c = (float3)(centroidsX[i], centroidsY[i], centroidsZ[i]);

    float nx = (c.x - cXMin) / dx;
    float ny = (c.y - cYMin) / dy;
    float nz = (c.z - cZMin) / dz;

    // Clamp to [0,1]
    nx = fmin(fmax(nx, 0.0f), 1.0f);
    ny = fmin(fmax(ny, 0.0f), 1.0f);
    nz = fmin(fmax(nz, 0.0f), 1.0f);

    codes[i] = morton3D(nx, ny, nz);
}