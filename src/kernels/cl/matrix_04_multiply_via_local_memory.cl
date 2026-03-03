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
    const uint loc_i = get_local_id(0);
    const uint loc_j = get_local_id(1);
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    __local float tile_a[GROUP_SIZE_X + 1][GROUP_SIZE_Y];
    __local float tile_b[GROUP_SIZE_X + 1][GROUP_SIZE_Y];

    c[j * w + i] = 0.0f;

    for (uint tile_k = 0; tile_k < k; tile_k += GROUP_SIZE_X) {
        const uint col_a = tile_k + loc_i;
        const uint row_b = tile_k / GROUP_SIZE_X * GROUP_SIZE_Y + loc_j;

        if (j < h && col_a < k) {
            tile_a[loc_i][loc_j] = a[j * k + col_a];
        } else {
            tile_a[loc_i][loc_j] = 0.0f;
        }

        if (i < w && row_b < k) {
            tile_b[loc_i][loc_j] = b[row_b * w + i];
        } else {
            tile_b[loc_i][loc_j] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint acc_idx = 0; acc_idx < GROUP_SIZE_X; ++acc_idx) {
            c[j * w + i] += tile_a[acc_idx][loc_j] * tile_b[loc_i][acc_idx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
