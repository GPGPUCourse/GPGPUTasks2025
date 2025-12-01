#pragma once

#include "../kernels/shared_structs/morton_code_gpu_shared.h"

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
unsigned int expandBits(unsigned int v)
{
    // Ensure we have only lowest 10 bits
    rassert(v == (v & 0x3FFu), 76389413321, v);

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
    unsigned int ix = std::min(std::max((int)(x * 1024.0f), 0), 1023);
    unsigned int iy = std::min(std::max((int)(y * 1024.0f), 0), 1023);
    unsigned int iz = std::min(std::max((int)(z * 1024.0f), 0), 1023);

    unsigned int xx = expandBits(ix);
    unsigned int yy = expandBits(iy);
    unsigned int zz = expandBits(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}
