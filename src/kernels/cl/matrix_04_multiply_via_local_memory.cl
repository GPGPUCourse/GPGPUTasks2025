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
    // TODO
    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);
    unsigned int x  = get_global_id(0);
    unsigned int y  = get_global_id(1);
    __local float data_a[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    __local float data_b[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    float acc = 0.0f;
    for (unsigned int i = 0; i < k; i += GROUP_SIZE_X) {

        unsigned int zA = i + lx;
        unsigned int zB = i + ly;

        data_a[ly][lx] = (y < h && zA < k) ? a[y * k + zA] : 0.0f;
        data_b[ly][lx] = (zB < k && x < w) ? b[zB * w + x] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < GROUP_SIZE_X; ++j) {
            acc += data_a[ly][j] * data_b[j][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = acc;
    }
}
