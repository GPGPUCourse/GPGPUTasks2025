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
    __local float local_a[GROUP_SIZE_X][GROUP_SIZE_Y];
    __local float local_b[GROUP_SIZE_X][GROUP_SIZE_Y];

    int x_global = get_global_id(1);
    int y_global = get_global_id(0);

    int x_local = get_local_id(1);
    int y_local = get_local_id(0);

    float acc = 0.0f;

    int tiles = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (int t = 0; t < tiles; t++) {
        int ax_global = x_global;
        int ay_global = t * GROUP_SIZE_X + y_local;
        int bx_global = t * GROUP_SIZE_Y + x_local;
        int by_global = y_global;

        if (ax_global < h && ay_global < k)
            local_a[x_local][y_local] = a[ax_global * k + ay_global];
        else
            local_a[x_local][y_local] = 0.0f;

        if (bx_global < k && by_global < w)
            local_b[x_local][y_local] = b[bx_global * w + by_global];
        else
            local_b[x_local][y_local] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < GROUP_SIZE_X; k++) {
            acc += local_a[x_local][k] * local_b[k][y_local];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x_global < h && y_global < w) {
        c[x_global * w + y_global] = acc;
    }
}
