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
    unsigned int id_x = get_global_id(0);
    unsigned int id_y = get_global_id(1);

    __local float a_tile[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float b_tile[GROUP_SIZE_X * GROUP_SIZE_Y];
    float res = 0;
    for (unsigned int i = 0; i * GROUP_SIZE_X < k; ++i) {
        a_tile[get_local_id(0) + get_local_id(1) * GROUP_SIZE_X] = a[i * GROUP_SIZE_X + id_y * k + get_local_id(0)];
        b_tile[get_local_id(0) + get_local_id(1) * GROUP_SIZE_X] = b[i * GROUP_SIZE_Y * w + id_x + get_local_id(1) * w];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < GROUP_SIZE_X; ++j) {
            res += a_tile[j + get_local_id(1) * GROUP_SIZE_X] * b_tile[get_local_id(0) + j * GROUP_SIZE_X];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[id_y * w + id_x] = res;
}
