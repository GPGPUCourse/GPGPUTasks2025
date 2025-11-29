#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#include "../shared_structs/camera_gpu_shared.h"
#include "geometry_helpers.cl"

#define R 5
#define DISTANCE_THRESHOLD 2.0f
#define DEPTH_THRESHOLD 0.15f
#define NORMAL_THRESHOLD 0.08f

static inline int pixel_index(const int i, const int j, const int width) {
    return i + j * width;
}

__kernel void denoise(
    __global float* framebuffer_in,
    __global float* framebuffer_out,
    __global float* depth_buffer,
    __global float* normal_buffer,
    __global const CameraViewGPU* camera
) {
    const int ii = get_global_id(0);
    const int jj = get_global_id(1);
    const int width = camera->K.width;
    const int height = camera->K.height;

    if (ii >= camera->K.width || jj >= camera->K.height) {
        return;
    }

    int index = pixel_index(ii, jj, width);
    float depth = depth_buffer[index];
    float3 normal = loadVertex(normal_buffer, index);

    float sum_v = 0;
    float sum_w = 0;

    for (int i = -R; i <= R; i++) {
        for (int j = -R; j <= R; j++) {
            const int ni = ii + i;
            const int nj = jj + j;
            if (ni >= camera->K.width || nj >= camera->K.height ||
                ni < 0 || nj < 0) continue;
            const int nindex = pixel_index(ni, nj, width);

            const float fi = (float) i;
            const float fj = (float) j;

            const float w_spatial = exp(-(fi*fi + fj*fj) / (2 * DISTANCE_THRESHOLD * DISTANCE_THRESHOLD));
            const float depth_difference = depth_buffer[nindex] - depth;
            const float w_depth = exp(-(depth_difference * depth_difference) / (2 * DEPTH_THRESHOLD * DEPTH_THRESHOLD));
            const float normal_dot_product = clamp(dot3(normal, loadVertex(normal_buffer, nindex)), 0.0f, 1.0f);
            const float w_normal = exp(-(1 - normal_dot_product * normal_dot_product) / (2 * NORMAL_THRESHOLD * NORMAL_THRESHOLD));
            const float w = w_spatial * w_depth * w_normal;
            const float v = framebuffer_in[nindex];
            sum_v += v * w;
            sum_w += w;
        }
    }

    framebuffer_out[index] = sum_v / sum_w;
}