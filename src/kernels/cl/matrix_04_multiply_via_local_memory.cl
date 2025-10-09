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
    __local float tmp_a[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float tmp_b[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float tmp_acc[GROUP_SIZE_X * GROUP_SIZE_Y];

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    int global_x = get_global_id(0);
    int global_y = get_global_id(1);

    unsigned int n = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    tmp_acc[local_y * GROUP_SIZE_X + local_x] = 0;

    for (unsigned int i = 0; i < n; i++) {
        unsigned int x = GROUP_SIZE_X * i + local_x;
        unsigned int y = global_y;

        if (x < k) {
            tmp_a[local_y * GROUP_SIZE_X + local_x] = a[y * k + x];
        } else {
            tmp_a[local_y * GROUP_SIZE_X + local_x] = 0;
        }

        x = global_x;
        y = GROUP_SIZE_X * i + local_y;
        if (y < k) {
            tmp_b[local_y * GROUP_SIZE_X + local_x] = b[y * w + x];
        } else {
            tmp_b[local_y * GROUP_SIZE_X + local_x] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < GROUP_SIZE_X; j++) {
            tmp_acc[local_y * GROUP_SIZE_X + local_x] += tmp_a[local_y * GROUP_SIZE_X + j] * tmp_b[j * GROUP_SIZE_X + local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_y * w + global_x] = tmp_acc[local_y * GROUP_SIZE_X + local_x];
}
