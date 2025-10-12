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
    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);
    __local float local_data_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float local_data_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    float acc = 0.0f;
    for (size_t ki = 0; ki < k; ki += GROUP_SIZE_X) {

        uint ki_a = local_x + ki;
        if (j < h && ki_a < k) {
            local_data_a[local_y][local_x] = a[j * k + ki_a];
        } else {
            local_data_a[local_y][local_x] = 0.0f;
        }

        uint ki_b = local_y + ki;
        if (i < w && ki_b < k) {
            local_data_b[local_y][local_x] = b[ki_b * w + i];
        } else {
            local_data_b[local_y][local_x] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t l_i = 0; l_i < GROUP_SIZE_X; ++l_i) {
            acc += local_data_a[local_y][l_i] * local_data_b[l_i][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i >= w || j >= h) {
        return;
    }

    c[j * w + i] = acc;
    
}
