#include <cfloat>
#include <device_launch_parameters.h>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>
#include <vector_types.h>

#include "../shared_structs/prim_gpu_cuda.h"
#include "../shared_structs/morton_code_gpu_shared.h"

namespace {

__device__ unsigned int expandBitsGpu(unsigned int v)
{
    // Ensure we have only lowest 10 bits
    // curassert(v == (v & 0x3FFu), 76389413321, v);

    // Magic bit expansion steps
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;

    return v;
}

// Convert 3D point in [0,1]^3 to 30-bit Morton code (10 bits per axis)
// Values outside [0,1] are clamped.
__device__ MortonCode morton3DGpu(float x, float y, float z)
{
    // Map and clamp to integer grid [0, 1023]
    unsigned int ix = umin(umax((unsigned int) (x * 1024.0f), 0), 1023u);
    unsigned int iy = umin(umax((unsigned int) (y * 1024.0f), 0), 1023u);
    unsigned int iz = umin(umax((unsigned int) (z * 1024.0f), 0), 1023u);

    unsigned int xx = expandBitsGpu(ix);
    unsigned int yy = expandBitsGpu(iy);
    unsigned int zz = expandBitsGpu(iz);

    // Interleave: x in bits [2,5,8,...], y in [1,4,7,...], z in [0,3,6,...]
    return (xx << 2) | (yy << 1) | zz;
}

}

/*
    // 2) Compute Morton codes for centroids (normalized to [0,1]^3)
    const float eps = 1e-9f;
    const float dx = std::max(cMax.x - cMin.x, eps);
    const float dy = std::max(cMax.y - cMin.y, eps);
    const float dz = std::max(cMax.z - cMin.z, eps);

    for (size_t i = 0; i < N; ++i) {
        const point3f& c = prims[i].centroid;
        float nx = (c.x - cMin.x) / dx;
        float ny = (c.y - cMin.y) / dy;
        float nz = (c.z - cMin.z) / dz;

        // Clamp to [0,1]
        nx = std::min(std::max(nx, 0.0f), 1.0f);
        ny = std::min(std::max(ny, 0.0f), 1.0f);
        nz = std::min(std::max(nz, 0.0f), 1.0f);

        prims[i].morton = morton3D(nx, ny, nz);
    }
*/

__global__ void set_morton_codes(
    unsigned int* data_morton,
    float3* data_centroid,
    const unsigned int nPrims,
    const float3* cMin,
    const float3* cMax
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nPrims) {
        return;
    }

    const float3 cMin_ref = *cMin;
    const float3 cMax_ref = *cMax;

    const float eps = 1e-9f;
    const float dx = fmaxf(cMax_ref.x - cMin_ref.x, eps);
    const float dy = fmaxf(cMax_ref.y - cMin_ref.y, eps);
    const float dz = fmaxf(cMax_ref.z - cMin_ref.z, eps);

    const float3& c = data_centroid[i];
    float nx = (c.x - cMin_ref.x) / dx;
    float ny = (c.y - cMin_ref.y) / dy;
    float nz = (c.z - cMin_ref.z) / dz;

    // Clamp to [0,1]
    nx = fminf(fmaxf(nx, 0.0f), 1.0f);
    ny = fminf(fmaxf(ny, 0.0f), 1.0f);
    nz = fminf(fmaxf(nz, 0.0f), 1.0f);

    data_morton[i] = morton3DGpu(nx, ny, nz);
}

namespace cuda {
void set_morton_codes(const gpu::WorkSize& workSize,
    gpu::shared_device_buffer_typed<unsigned int>& data_morton,
    gpu::shared_device_buffer_typed<float3>& data_centroid,
    const unsigned int nPrims,
    const gpu::shared_device_buffer_typed<float3>& cMin,
    const gpu::shared_device_buffer_typed<float3>& cMax)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::set_morton_codes<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        data_morton.cuptr(),
        data_centroid.cuptr(),
        nPrims,
        cMin.cuptr(),
        cMax.cuptr()
    );
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
