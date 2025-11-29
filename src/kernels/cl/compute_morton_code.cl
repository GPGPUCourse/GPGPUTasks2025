#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"

// Helper: expand 10 bits into 30 bits by inserting 2 zeros between each bit
static inline unsigned int expandBits(unsigned int v)
{
    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

static inline uint morton3D(float x, float y, float z)
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

__kernel void compute_morton_code(
    __global const float* input_centroids,
    __global const float* input_bounds,
    __global       uint* output_mix,
                   int  nfaces)
{
    const unsigned int i = get_global_id(0);
    const float eps = 1e-9f;
    if (i >= nfaces) {
        return;
    }
    output_mix[2 * i + 1] = i;
    const float dx = max(input_bounds[3] - input_bounds[0], eps);
    const float dy = max(input_bounds[4] - input_bounds[1], eps);
    const float dz = max(input_bounds[5] - input_bounds[2], eps);
    float nx = (input_centroids[3 * i] - input_bounds[0]) / dx;
    float ny = (input_centroids[3 * i + 1] - input_bounds[1]) / dy;
    float nz = (input_centroids[3 * i + 2] - input_bounds[2]) / dz;
    nx = min(max(nx, 0.0f), 1.0f);
    ny = min(max(ny, 0.0f), 1.0f);
    nz = min(max(nz, 0.0f), 1.0f);
    output_mix[2 * i] = morton3D(nx, ny, nz);
}
