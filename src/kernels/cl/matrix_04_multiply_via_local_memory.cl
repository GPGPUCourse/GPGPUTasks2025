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
    const unsigned int lx =  get_local_id(0);
    const unsigned int ly =  get_local_id(1);
    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);

    __local float a_tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float b_tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    float sum = 0.0f;
    for (unsigned int t = 0; t < k; t += GROUP_SIZE_X) {
        if (row < h && t + lx < k) {
            a_tile[ly][lx] = a[k * row + (t + lx)];
        } else {
            a_tile[ly][lx] = 0.0;
        }

        if (t + ly < k && col < w) {
            b_tile[ly][lx] = b[w * (t + ly) + col];
        } else {
            b_tile[ly][lx] = 0.0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int xx = 0; xx < GROUP_SIZE_X; xx++) {
            sum += a_tile[ly][xx] * b_tile[xx][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

    }

    if (col < w && row < h) {
        c[row *w + col] = sum;
    }
}
