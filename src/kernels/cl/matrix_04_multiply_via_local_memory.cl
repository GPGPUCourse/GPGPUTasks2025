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
    const uint global_i = get_global_id(0);
    const uint global_j = get_global_id(1);

    __local float prod_a[GROUP_SIZE_X][GROUP_SIZE_Y], prod_b[GROUP_SIZE_X][GROUP_SIZE_Y];

    const uint local_i = get_local_id(0);
    const uint local_j = get_local_id(1);

    float res = 0;

    for (uint t = 0; t < k; t += GROUP_SIZE_X) {
        const uint local_a_column = t + local_i;
        const uint local_a_row = global_j;

        const uint local_b_column = global_i;
        const uint local_b_row = t + local_j;

        prod_a[local_i][local_j] = a[local_a_row * k + local_a_column];
        prod_b[local_i][local_j] = b[local_b_row * w + local_b_column];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint tt = 0; tt < GROUP_SIZE_X; tt++) {
            res += prod_a[tt][local_j] * prod_b[local_i][tt];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_j * w + global_i] = res;
}
