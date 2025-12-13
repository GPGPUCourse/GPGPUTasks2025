#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/build_bvh.h"
#include "cl/generated_kernels/calc_centroids_aabb.h"
#include "cl/generated_kernels/calc_morton.h"
#include "cl/generated_kernels/copy_array.h"
#include "cl/generated_kernels/get_sorted_morton_codes.h"
#include "cl/generated_kernels/max_reduction.h"
#include "cl/generated_kernels/merge_sort.h"
#include "cl/generated_kernels/min_reduction.h"
#include "cl/generated_kernels/ray_tracing_render_brute_force.h"
#include "cl/generated_kernels/ray_tracing_render_using_lbvh.h"

#include "libgpu/opencl/engine.h"
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
void ray_tracing_render_brute_force(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    gpu::gpu_mem_32i& framebuffer_face_id,
    gpu::gpu_mem_32f& framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void ray_tracing_render_using_lbvh(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU>& bvhNodes, const gpu::gpu_mem_32u& leafTriIndices,
    gpu::gpu_mem_32i& framebuffer_face_id,
    gpu::gpu_mem_32f& framebuffer_ambient_occlusion,
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
const ProgramBinaries& getCalcCentroidsAABB()
{
    return opencl_binaries_calc_centroids_aabb;
}
const ProgramBinaries& getCalcMorton()
{
    return opencl_binaries_calc_morton;
}
const ProgramBinaries& getCopyArray()
{
    return opencl_binaries_copy_array;
}
const ProgramBinaries& getMaxReduction()
{
    return opencl_binaries_max_reduction;
}
const ProgramBinaries& getMergeSort()
{
    return opencl_binaries_merge_sort;
}
const ProgramBinaries& getMinReduction()
{
    return opencl_binaries_min_reduction;
}
const ProgramBinaries& getGetSortedMortonCodes()
{
    return opencl_binaries_get_sorted_morton_codes;
}
const ProgramBinaries& getBuildBVH()
{
    return opencl_binaries_build_bvh;
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
