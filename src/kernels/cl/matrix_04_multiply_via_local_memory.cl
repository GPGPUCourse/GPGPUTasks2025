#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define TILE_W GROUP_SIZE_X
#define TILE_H GROUP_SIZE_Y
#define TILE_K 16

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
    const uint x = get_group_id(0) * TILE_W + local_x;
    const uint y = get_group_id(1) * TILE_H + local_y;

    if (x >= w || y >= h) {
        return;
    }

    float sum = 0;
    __local float local_data_A[TILE_H][TILE_K + 1];
    __local float local_data_B[TILE_K][TILE_W + 1];
    for (uint i = 0; i < k; i += TILE_K) {
        uint a_column = i + local_x;
        if (y < h && a_column < k) {
            local_data_A[local_y][local_x] = a[y * k + a_column];
        } else {
            local_data_A[local_y][local_x] = 0;
        }

        uint b_row = i + local_y;
        if (b_row < k && x < w) {
            local_data_B[local_y][local_x] = b[b_row * w + x];
        } else {
            local_data_B[local_y][local_x] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint j = 0; j < TILE_K; ++j) {
            sum += local_data_A[local_y][j] * local_data_B[j][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[y * w + x] = sum;
}
