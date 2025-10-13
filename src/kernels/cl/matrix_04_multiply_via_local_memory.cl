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
    __local float local_data_a[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float local_data_b[GROUP_SIZE_X * GROUP_SIZE_Y];

    const unsigned int global_y = get_global_id(0);
    const unsigned int global_x = get_global_id(1);
    const unsigned int global_index = global_x * w + global_y;

    const unsigned int local_y = get_local_id(0);
    const unsigned int local_x = get_local_id(1);
    const unsigned int local_index = local_x * GROUP_SIZE_X + local_y;

    int global_a_index;
    int global_b_index;
    int z;

    float res = 0;
    for (int z_shift = 0; z_shift < k; z_shift += GROUP_SIZE_X) {
        global_a_index = global_x * k + (z_shift + local_y);
        global_b_index = (z_shift + local_x) * w + global_y;

        if (global_a_index < h * k) {
            local_data_a[local_index] = a[global_a_index];
        }
        if (global_b_index < k * w) {
            local_data_b[local_index] = b[global_b_index];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (global_index < h * w) {
            for (int dz = 0; dz < GROUP_SIZE_X; dz++) {
                res += local_data_a[local_x * GROUP_SIZE_X + dz] * local_data_b[dz * GROUP_SIZE_X + local_y];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[global_index] = res;
}
