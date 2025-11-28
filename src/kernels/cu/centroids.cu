#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "geometry_helpers.cu"


__global__ void centroid(
    const float* vertices, const unsigned int* faces,
    unsigned int nfaces, float* centroids_x,
    float* centroids_y, float* centroids_z)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nfaces) return;

    uint3 f = loadFace(faces, index);
    float3 a = loadVertex(vertices, f.x);
    float3 b = loadVertex(vertices, f.y);
    float3 c = loadVertex(vertices, f.z);

    centroids_x[index] = (a.x + b.x + c.x) / 3.0f;
    centroids_y[index] = (a.y + b.y + c.y) / 3.0f;
    centroids_z[index] = (a.z + b.z + c.z) / 3.0f;
}

namespace cuda {
void centroid(const gpu::WorkSize &workSize, const gpu::gpu_mem_32f &vertices,
    const gpu::gpu_mem_32u& faces, unsigned int nfaces,
    gpu::gpu_mem_32f &centroids_x, gpu::gpu_mem_32f &centroids_y, gpu::gpu_mem_32f &centroids_z)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::centroid<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(vertices.cuptr(),
        faces.cuptr(), nfaces, centroids_x.cuptr(),
        centroids_y.cuptr(), centroids_z.cuptr());
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
