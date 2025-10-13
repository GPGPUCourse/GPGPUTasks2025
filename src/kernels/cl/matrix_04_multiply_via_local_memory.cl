#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

inline void load_local_a(__global const float* a, local float* local_a,
                         uint i, uint lx, uint ly, uint gy,
                         uint k, uint h) {
    const uint ax = i + lx;
    const uint ay = gy * GROUP_SIZE_Y + ly;
    if (ax >= k || ay >= h) {
        local_a[ly * GROUP_SIZE_X + lx] = 0.0f;
    } else {
        local_a[ly * GROUP_SIZE_X + lx] = a[ay * k + ax];
    }
}

inline void load_local_b(__global const float* b, local float* local_b,
                         uint i, uint lx, uint ly, uint gx,
                         uint w, uint k) {
    const uint by = i + ly;
    const uint bx = gx * GROUP_SIZE_X + lx;
    if (bx >= w || by >= k) {
        local_b[ly * GROUP_SIZE_X + lx] = 0.0f;
    } else {
        local_b[ly * GROUP_SIZE_X + lx] = b[by * w + bx];
    }
}

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    // TODO
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if (x >= w || y >= h) {
        return;
    }

    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);
    const uint gx = get_group_id(0);
    const uint gy = get_group_id(1);

    local float local_a[GROUP_SIZE_X * GROUP_SIZE_Y];
    local float local_b[GROUP_SIZE_X * GROUP_SIZE_Y];

    float value = 0;

    for (int i = 0; i < k; i += GROUP_SIZE_X) {
        load_local_a(a, local_a, i, lx, ly, gy, k, h);
        load_local_b(b, local_b, i, lx, ly, gx, w, k);

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < GROUP_SIZE_X && (i + j) < k; j++) {
            value += local_b[j * GROUP_SIZE_X + lx] * local_a[ly * GROUP_SIZE_X + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[y * w + x] = value;
}