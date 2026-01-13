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
    __local float Asub[GROUP_SIZE_X][GROUP_SIZE_Y];
    __local float Bsub[GROUP_SIZE_X][GROUP_SIZE_Y];

    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);

    float sum = 0.0f;

    for (unsigned int t = 0; t < k; t += GROUP_SIZE_X) {
        if ((y < h) && (t + local_x < k)) {
            Asub[local_y][local_x] = a[y * k + (t + local_x)];
        } else {
            Asub[local_y][local_x] = 0.0f;
        }

        if ((x < w) && (t + local_y < k)) {
            Bsub[local_y][local_x] = b[(local_y + t) * w + x];
        } else {
            Bsub[local_y][local_x] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int l = 0; l < GROUP_SIZE_X; l++) {
            sum += Asub[local_y][l] * Bsub[l][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ((y < h) && (x < w)) {
        c[y * w + x] = sum;
    }
}
