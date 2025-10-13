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
    unsigned int global_x = get_global_id(0);
    unsigned int global_y = get_global_id(1);
    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);

    __local float local_mem_1[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float local_mem_2[GROUP_SIZE_Y][GROUP_SIZE_X];

    float res = 0.f;

    if (global_x >= w || global_y >= h)
        return;

    for (int p = 0; p < k; p += GROUP_SIZE_X) {
        unsigned int p_1 = p + local_x;
        unsigned int p_2 = p + local_y;

        if (global_y < h && p_1 < k)
            local_mem_1[local_y][local_x] = a[global_y * k + p_1];
        else
            local_mem_1[local_y][local_x] = 0.f;

        if (global_x < w && p_2 < k)
            local_mem_2[local_y][local_x] = b[global_x + p_2 * w];
        else
            local_mem_2[local_y][local_x] = 0.f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < GROUP_SIZE_X; ++t)
            res += local_mem_1[local_y][t] * local_mem_2[t][local_x];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_y * w + global_x] = res;

}
