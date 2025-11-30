#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/build_lbvh_skeleton.h"
#include "cl/generated_kernels/compute_morton_codes.h"
#include "cl/generated_kernels/fill_index.h"
#include "cl/generated_kernels/fill_with_value.h"
#include "cl/generated_kernels/populate_aabb_over_lbvh.h"
#include "cl/generated_kernels/radix_sort_bucket_prefix.h"
#include "cl/generated_kernels/radix_sort_local_counting.h"
#include "cl/generated_kernels/radix_sort_prefix_tiled.h"
#include "cl/generated_kernels/radix_sort_scatter.h"
#include "cl/generated_kernels/ray_tracing_render_brute_force.h"
#include "cl/generated_kernels/ray_tracing_render_using_lbvh.h"

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
const ProgramBinaries& getComputeMortonCodes()
{
    return opencl_binaries_compute_morton_codes;
}
const ProgramBinaries& getBuildLVBHSkeleton()
{
    return opencl_binaries_build_lbvh_skeleton;
}
const ProgramBinaries& getFillWithValue()
{
    return opencl_binaries_fill_with_value;
}
const ProgramBinaries& getPopulateAABBOverLBVH()
{
    return opencl_binaries_populate_aabb_over_lbvh;
}
const ProgramBinaries& getFillIndex()
{
    return opencl_binaries_fill_index;
}
const ProgramBinaries& getRadixSortBucketPrefix()
{
    return opencl_binaries_radix_sort_bucket_prefix;
}
const ProgramBinaries& getRadixSortLocalCounting()
{
    return opencl_binaries_radix_sort_local_counting;
}
const ProgramBinaries& getRadixSortPrefixTiled()
{
    return opencl_binaries_radix_sort_prefix_tiled;
}
const ProgramBinaries& getRadixSortScatter()
{
    return opencl_binaries_radix_sort_scatter;
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
