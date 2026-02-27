#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

unsigned int div_ceil(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);
    if (i >= h || j >= w) {
        return;
    }

    const unsigned int l_j = get_local_id(0);
    const unsigned int l_i = get_local_id(1);

    __local float a_tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float b_tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    const unsigned int groups_count = div_ceil(w, (unsigned int)GROUP_SIZE_X);

    float sum = 0;
    for (unsigned int group = 0; group < groups_count; group++) {
        int a_i = i;
        int a_j = group * GROUP_SIZE_X + l_j;
        a_tile[l_i][l_j] = a_j < k ? a[a_i * k + a_j] : 0;

        int b_i = group * GROUP_SIZE_Y + l_i;
        int b_j = j;
        b_tile[l_i][l_j] = b_i < k ? b[b_i * w + b_j] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int q = 0; q < GROUP_SIZE_X; q++) {
            sum += a_tile[l_i][q] * b_tile[q][l_j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[i * w + j] = sum;
}
