
#include "helpers/rassert.cl"
#include "../shared_structs/morton_code_gpu_shared.h"
#include "../shared_structs/camera_gpu_shared.h"
#include "../shared_structs/bvh_node_gpu_shared.h"
#include "../shared_structs/aabb_gpu_shared.h"

#include "camera_helpers.cl"
#include "geometry_helpers.cl"
#include "random_helpers.cl"
#include "../defines.h"

static inline void process_pixel(
    float* sum_brightness,
    int* cnt_neighbors,
    __global const int*              framebuffer_face_id,
    __global const float*            framebuffer_ambient_occlusion,
    __global const CameraViewGPU* camera,
             int expected_face_id,
             int i,
             int j
)
{
    if (i < 0 || j < 0 || i >= camera->K.width || j >= camera->K.height) {
        return;
    }

    const uint idx = j * camera->K.width + i;
    if (framebuffer_face_id[idx] != expected_face_id) {
        return;
    }
    *sum_brightness += framebuffer_ambient_occlusion[idx];
    *cnt_neighbors += 1;
}

__kernel void denoise_facets(
    __global int*              framebuffer_face_id,
    __global float*            framebuffer_ambient_occlusion,
    __global const CameraViewGPU* camera)
{
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    rassert(camera.magic_bits_guard == CAMERA_VIEW_GPU_MAGIC_BITS_GUARD, 786435342);
    if (i >= camera->K.width || j >= camera->K.height)
        return;

    const uint idx = j * camera->K.width + i;
    float sum = 0.0;
    int cnt = 0;
    int expected = framebuffer_face_id[idx];
    if (expected == -1) {
        return;
    }
    process_pixel(&sum, &cnt, framebuffer_face_id, framebuffer_ambient_occlusion, camera, expected, i, j);
    process_pixel(&sum, &cnt, framebuffer_face_id, framebuffer_ambient_occlusion, camera, expected, i-1, j);
    process_pixel(&sum, &cnt, framebuffer_face_id, framebuffer_ambient_occlusion, camera, expected, i+1, j);
    process_pixel(&sum, &cnt, framebuffer_face_id, framebuffer_ambient_occlusion, camera, expected, i, j-1);
    process_pixel(&sum, &cnt, framebuffer_face_id, framebuffer_ambient_occlusion, camera, expected, i, j+1);

    process_pixel(&sum, &cnt, framebuffer_face_id, framebuffer_ambient_occlusion, camera, expected, i-1, j-1);
    process_pixel(&sum, &cnt, framebuffer_face_id, framebuffer_ambient_occlusion, camera, expected, i+1, j-1);
    process_pixel(&sum, &cnt, framebuffer_face_id, framebuffer_ambient_occlusion, camera, expected, i-1, j+1);
    process_pixel(&sum, &cnt, framebuffer_face_id, framebuffer_ambient_occlusion, camera, expected, i+1, j+1);

    sum = sum / (float)cnt;
    framebuffer_ambient_occlusion[idx] = sum;
}
