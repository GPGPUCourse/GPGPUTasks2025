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
    // DONE

    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    unsigned int gx = get_group_id(0) * GROUP_SIZE_X + lx;
    unsigned int gy = get_group_id(1) * GROUP_SIZE_Y + ly;

    __local float tileA[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tileB[GROUP_SIZE_Y][GROUP_SIZE_X];

    float sum = 0.0f;

    unsigned int numTiles = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (unsigned int t = 0; t < numTiles; ++t) {
        unsigned int tiledA_col = t * GROUP_SIZE_X + lx;
        unsigned int tiledB_row = t * GROUP_SIZE_Y + ly;

        if (gy < h && tiledA_col < k) {
            tileA[ly][lx] = a[gy * k + tiledA_col];
        } else {
            tileA[ly][lx] = 0.0f;
        }

        if (gx < w && tiledB_row < k) {
            tileB[ly][lx] = b[tiledB_row * w + gx];
        } else {
            tileB[ly][lx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i) {
            sum += tileA[ly][i] * tileB[i][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gx < w && gy < h) {
        c[gy * w + gx] = sum;
    }
}
