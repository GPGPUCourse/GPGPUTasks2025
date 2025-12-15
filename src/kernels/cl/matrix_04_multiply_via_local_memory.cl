#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);
    const unsigned int gx = get_group_id(0);
    const unsigned int gy = get_group_id(1);

    const unsigned int x = gx * GROUP_SIZE_X + lx;
    const unsigned int y = gy * GROUP_SIZE_Y + ly;

    __local float tileA[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tileB[GROUP_SIZE_Y][GROUP_SIZE_X];

    float acc = 0.0f;

    for (unsigned int kk = 0; kk < k; kk += GROUP_SIZE_X) {
        const unsigned int a_col = kk + lx;
        const unsigned int b_row = kk + ly;

        if (y < h && a_col < k) {
            tileA[ly][lx] = a[y * k + a_col];
        } else {
            tileA[ly][lx] = 0.0f;
        }

        if (b_row < k && x < w) {
            tileB[ly][lx] = b[b_row * w + x];
        } else {
            tileB[ly][lx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int inner = 0; inner < GROUP_SIZE_X && (kk + inner) < k; ++inner) {
            acc += tileA[ly][inner] * tileB[inner][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = acc;
    }
}
