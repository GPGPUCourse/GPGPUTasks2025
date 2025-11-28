#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "../shared_structs/morton_code_gpu_shared.h"

__device__ __forceinline__ unsigned int expandBits(unsigned int v)
{
    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
__device__ __forceinline__ MortonCode morton3D(float x, float y, float z)
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

__global__ void morton_code(
    const float* centroids_x,
    const float* centroids_y,
    const float* centroids_z,
    MortonCode* morton_codes,
    unsigned int* indices,
    const unsigned int n,
    float mn_x, float mn_y, float mn_z,
    float mx_x, float mx_y, float mx_z)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    float x = (centroids_x[index] - mn_x) / (mx_x - mn_x);
    float y = (centroids_y[index] - mn_y) / (mx_y - mn_y);
    float z = (centroids_z[index] - mn_z) / (mx_z - mn_z);

    morton_codes[index] = morton3D(x, y, z);
    indices[index] = index;
}

namespace cuda {
void morton_code(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &centroids_x, const gpu::gpu_mem_32f &centroids_y,
    const gpu::gpu_mem_32f &centroids_z, gpu::gpu_mem_32u &morton_codes,
    gpu::gpu_mem_32u &indices, unsigned int n, float mn_x, float mn_y, float mn_z,
    float mx_x, float mx_y, float mx_z)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::morton_code<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(centroids_x.cuptr(),
        centroids_y.cuptr(), centroids_z.cuptr(), morton_codes.cuptr(),
        indices.cuptr(), n, mn_x, mn_y, mn_z, mx_x, mx_y, mx_z);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
