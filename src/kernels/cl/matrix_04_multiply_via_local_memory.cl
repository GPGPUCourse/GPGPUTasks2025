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
    __local float A_tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float B_tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    size_t l_x = get_local_id(0);
    size_t l_y = get_local_id(1);

    size_t x = get_group_id(0) * GROUP_SIZE_X + l_x;
    size_t y = get_group_id(1) * GROUP_SIZE_Y + l_y;

    float res = 0.;

    for (size_t ki = 0; ki < k; ki+=GROUP_SIZE_X) {
        if (y < h && ki + l_x < k) {
            A_tile[l_y][l_x] = a[y * k + ki + l_x];
        } else {
            A_tile[l_y][l_x] = 0.0f;
        }

        if (x < w && ki + l_y < k) {
            B_tile[l_y][l_x] = b[(ki + l_y) * w + x];
        } else {
            B_tile[l_y][l_x] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < GROUP_SIZE_X; ++i) {
            res += A_tile[l_y][i] * B_tile[i][l_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

    }

    if (x < w && y < h) {
        c[y * w + x] = res;
    }
}
