#pragma once

#include <libgpu/vulkan/engine.h>

#include "shared_structs/camera_gpu_shared.h"
#include "shared_structs/bvh_node_gpu_shared.h"
#include "shared_structs/morton_code_gpu_shared.h"

namespace cuda {
void aplusb(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n);

void ray_tracing_render_brute_force(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces);
void ray_tracing_render_using_lbvh(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &vertices, const gpu::gpu_mem_32u &faces,
    const gpu::shared_device_buffer_typed<BVHNodeGPU> &bvhNodes, const gpu::gpu_mem_32u &leafTriIndices,
    gpu::gpu_mem_32i &framebuffer_face_id,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion,
    gpu::shared_device_buffer_typed<CameraViewGPU> camera,
    unsigned int nfaces);

void min_array(const gpu::WorkSize &workSize, const gpu::gpu_mem_32f &a,
    const gpu::gpu_mem_32f &b, const gpu::gpu_mem_32f &c,
    unsigned int n, gpu::gpu_mem_32f &out_a, gpu::gpu_mem_32f &out_b, gpu::gpu_mem_32f &out_c);
void max_array(const gpu::WorkSize &workSize, const gpu::gpu_mem_32f &a,
    const gpu::gpu_mem_32f &b, const gpu::gpu_mem_32f &c,
    unsigned int n, gpu::gpu_mem_32f &out_a, gpu::gpu_mem_32f &out_b, gpu::gpu_mem_32f &out_c);

void centroid(const gpu::WorkSize &workSize, const gpu::gpu_mem_32f &vertices,
    const gpu::gpu_mem_32u& faces, unsigned int nfaces,
    gpu::gpu_mem_32f &centroids_x, gpu::gpu_mem_32f &centroids_y, gpu::gpu_mem_32f &centroids_z);

void morton_code(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32f &centroids_x, const gpu::gpu_mem_32f &centroids_y,
    const gpu::gpu_mem_32f &centroids_z, gpu::gpu_mem_32u &morton_codes,
    gpu::gpu_mem_32u &indices, unsigned int n, float mn_x, float mn_y, float mn_z,
    float mx_x, float mx_y, float mx_z);

void merge_sort(const gpu::WorkSize &workSize,
        const gpu::gpu_mem_32u &indices, const gpu::gpu_mem_32u &morton_codes,
        gpu::gpu_mem_32u& out_morton_codes, gpu::gpu_mem_32u &out_indices,
        unsigned int sorted_k, unsigned int n);

void make_lbvh(const gpu::WorkSize &workSize, const gpu::gpu_mem_32u &codes,
        const gpu::gpu_mem_32u &leaf_indices,
        const gpu::gpu_mem_32f &vertices,
        const gpu::gpu_mem_32u &faces,
        unsigned int nfaces,
        gpu::shared_device_buffer_typed<BVHNodeGPU> &bvh_nodes,
        gpu::gpu_mem_32i &indices_up);

void update_aabb(const gpu::WorkSize &workSize,
        gpu::shared_device_buffer_typed<BVHNodeGPU> &nodes,
        const gpu::gpu_mem_32i &indices_up,
        unsigned int n);
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
