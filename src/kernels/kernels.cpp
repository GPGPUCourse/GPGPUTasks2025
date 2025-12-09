#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/ray_tracing_render_brute_force.h"
#include "cl/generated_kernels/ray_tracing_render_using_lbvh.h"
#include "cl/generated_kernels/calculate_centroid_bounds.h"
#include "cl/generated_kernels/init_triangle_codes.h"
#include "cl/generated_kernels/merge_sort_morton.h"
#include "cl/generated_kernels/build_lbvh.h"
#include "cl/generated_kernels/wave_calculate_aabb.h"
#include "cl/generated_kernels/denoise_facets.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/ray_tracing_render_brute_force_comp.h"
#include "vk/generated_kernels/ray_tracing_render_using_lbvh_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void ray_tracing_render_brute_force(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void ray_tracing_render_using_lbvh(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU> &bvhNodes, const gpu::gpu_mem_32u &leafTriIndices,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
} // namespace cuda
#endif

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}

const ProgramBinaries& getRTBruteForce()
{
    return opencl_binaries_ray_tracing_render_brute_force;
}

const ProgramBinaries& getRTWithLBVH()
{
    return opencl_binaries_ray_tracing_render_using_lbvh;
}

const ProgramBinaries& getCalcCentroidBounds()
{
    return opencl_binaries_calculate_centroid_bounds;
}
const ProgramBinaries& getInitTriangleCodes()
{
    return opencl_binaries_init_triangle_codes;
}
const ProgramBinaries& getMergeSortMorton()
{
    return opencl_binaries_merge_sort_morton;
}
const ProgramBinaries& getBuildLBVH()
{
    return opencl_binaries_build_lbvh;
}
const ProgramBinaries& getWaveCalculateAABB()
{
    return opencl_binaries_wave_calculate_aabb;
}
const ProgramBinaries& getDenoiseFacets()
{
    return opencl_binaries_denoise_facets;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}

const ProgramBinaries& getRTBruteForce()
{
    return vulkan_binaries_ray_tracing_render_brute_force_comp;
}

const ProgramBinaries& getRTWithLBVH()
{
    return vulkan_binaries_ray_tracing_render_using_lbvh_comp;
}
} // namespace avk2
