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
    __local float local_data_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float local_data_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    const uint index_x = get_global_id(0);
    const uint index_y = get_global_id(1);
    const uint local_index_y = get_local_id(1);
    const uint local_index_x = get_local_id(0);

    float sum = 0;

    for (int t = 0; t < (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X; ++t) {
        int index_x_a = local_index_x + t * GROUP_SIZE_X;
        int index_y_a = index_y;

        int index_x_b = index_x;
        int index_y_b = local_index_y + t * GROUP_SIZE_Y;

        if (index_y_a < h && index_x_a < k) {
            local_data_a[local_index_y][local_index_x] = a[index_x_a + index_y_a * k];
        }

        if (index_y_b < k && index_x_b < w) {
            local_data_b[local_index_y][local_index_x] = b[index_x_b + index_y_b * w];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < GROUP_SIZE_X; ++i) {
            sum += local_data_a[local_index_y][i] * local_data_b[i][local_index_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index_y < h && index_x < w) {
        c[index_x + index_y * w] = sum;
    }
}
