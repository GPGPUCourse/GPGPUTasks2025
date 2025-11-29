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

static void inline atomic_min_float(__global float* ptr, float new_value) {
    float cur = *ptr;
    while (new_value < cur) {
        new_value = atomic_xchg(ptr, min(cur, new_value));
        cur = *ptr;
    }
}

static void inline atomic_max_float(__global float* ptr, float new_value) {
    float cur;
    while (new_value > (cur = *ptr)) {
        new_value = atomic_xchg(ptr, max(cur, new_value));
    }
}

__kernel void compute_centroids(
    __global const uint* input_faces,
    __global const float* input_vertices,
    __global       float* output_centroids,
    __global       float* output_bounds,
                   int  nfaces)
{
    const unsigned int i = get_global_id(0);
    if (i >= nfaces) {
        return;
    }
    uint faceX = input_faces[i * 3];
    uint faceY = input_faces[i * 3 + 1];
    uint faceZ = input_faces[i * 3 + 2];
    float xx = input_vertices[faceX * 3];
    float xy = input_vertices[faceX * 3 + 1];
    float xz = input_vertices[faceX * 3 + 2];
    float yx = input_vertices[faceY * 3];
    float yy = input_vertices[faceY * 3 + 1];
    float yz = input_vertices[faceY * 3 + 2];
    float zx = input_vertices[faceZ * 3];
    float zy = input_vertices[faceZ * 3 + 1];
    float zz = input_vertices[faceZ * 3 + 2];
    output_centroids[3 * i] = (xx + yx + zx) * (1.f / 3.f);
    output_centroids[3 * i + 1] = (xy + yy + zy) * (1.f / 3.f);
    output_centroids[3 * i + 2] = (xz + yz + zz) * (1.f / 3.f);
    atomic_min_float(output_bounds, output_centroids[3 * i]);
    atomic_min_float(output_bounds + 1, output_centroids[3 * i + 1]);
    atomic_min_float(output_bounds + 2, output_centroids[3 * i + 2]);
    atomic_max_float(output_bounds + 3, output_centroids[3 * i]);
    atomic_max_float(output_bounds + 4, output_centroids[3 * i + 1]);
    atomic_max_float(output_bounds + 5, output_centroids[3 * i + 2]);
}
