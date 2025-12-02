#pragma once

#include <libgpu/vulkan/engine.h>

#include "shared_structs/bvh_node_gpu_shared.h"
#include "shared_structs/camera_gpu_shared.h"
#include "shared_structs/morton_code_gpu_shared.h"

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void ray_tracing_render_brute_force(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    gpu::gpu_mem_32i& framebuffer_face_id,
    gpu::gpu_mem_32f& framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces);
void ray_tracing_render_using_lbvh(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices, const gpu::gpu_mem_32u& faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU>& bvhNodes, const gpu::gpu_mem_32u& leafTriIndices,
    gpu::gpu_mem_32i& framebuffer_face_id,
    gpu::gpu_mem_32f& framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces);

// unsigned int triIndex;
// unsigned int morton;
// AABBGPU aabb;
// float3 centroid;

void merge_sort(
    const gpu::WorkSize& workSize,
    const gpu::shared_device_buffer_typed<unsigned int>& input_data_triIndex,
    const gpu::shared_device_buffer_typed<unsigned int>& input_data_morton,
    const gpu::shared_device_buffer_typed<AABBGPU>& input_data_aabb,
    const gpu::shared_device_buffer_typed<float3>& input_data_centroid,
    gpu::shared_device_buffer_typed<unsigned int>& output_data_triIndex,
    gpu::shared_device_buffer_typed<unsigned int>& output_data_morton,
    gpu::shared_device_buffer_typed<AABBGPU>& output_data_aabb,
    gpu::shared_device_buffer_typed<float3>& output_data_centroid,
    int sorted_k,
    int n);

void build_prim(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& vertices,
    const gpu::gpu_mem_32u& faces,
    const int nVertices,
    const int nFaces,
    gpu::shared_device_buffer_typed<unsigned int>& output_data_triIndex,
    gpu::shared_device_buffer_typed<unsigned int>& output_data_morton,
    gpu::shared_device_buffer_typed<AABBGPU>& output_data_aabb,
    gpu::shared_device_buffer_typed<float3>& output_data_centroid,
    gpu::shared_device_buffer_typed<float3>& cMin,
    gpu::shared_device_buffer_typed<float3>& cMax);

void init_min_max(gpu::shared_device_buffer_typed<float3>& cMin,
    gpu::shared_device_buffer_typed<float3>& cMax);

void set_morton_codes(const gpu::WorkSize& workSize,
    gpu::shared_device_buffer_typed<unsigned int>& data_morton,
    gpu::shared_device_buffer_typed<float3>& data_centroid,
    const unsigned int nPrims,
    const gpu::shared_device_buffer_typed<float3>& cMin,
    const gpu::shared_device_buffer_typed<float3>& cMax);

void pre_build_bvh(const gpu::WorkSize& workSize,
    const int n,
    gpu::shared_device_buffer_typed<unsigned int>& data_triIndex,
    gpu::shared_device_buffer_typed<MortonCode>& data_morton,
    gpu::shared_device_buffer_typed<AABBGPU>& data_aabb,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& outNodes,
    gpu::shared_device_buffer_typed<int>& parentIndices);

void fill_zeros(const gpu::WorkSize& workSize,
    gpu::shared_device_buffer_typed<int>& data,
    const int n);

void build_bvh(const gpu::WorkSize& workSize,
    const int nLeaves,
    gpu::shared_device_buffer_typed<BVHNodeGPU>& nodes,
    gpu::shared_device_buffer_typed<int>& parentIndices,
    gpu::shared_device_buffer_typed<int>& atomicCounters);
}

namespace ocl {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getRTBruteForce();
const ProgramBinaries& getRTWithLBVH();
}

namespace avk2 {
const ProgramBinaries& getAplusB();

const ProgramBinaries& getRTBruteForce();
const ProgramBinaries& getRTWithLBVH();
}
