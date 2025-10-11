#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define TILE_SIZE GROUP_SIZE_X

inline void load_tile_A(
    __global const float* a,
    __local float* A_tiles,
    unsigned int row,
    unsigned int k,
    unsigned int tile_index,
    unsigned int local_row,
    unsigned int local_col)
{
    A_tiles[local_row * GROUP_SIZE_X + local_col] = a[row * k + (tile_index * TILE_SIZE + local_col)];
}

inline void load_tile_B(
    __global const float* b,
    __local float* B_tiles,
    unsigned int col,
    unsigned int w,
    unsigned int tile_index,
    unsigned int local_row,
    unsigned int local_col)
{
    B_tiles[local_row * GROUP_SIZE_X + local_col] = b[(tile_index * TILE_SIZE + local_row) * w + col];
}

inline float compute_tile_product(
    __local float* As,
    __local float* Bs,
    unsigned int local_row,
    unsigned int local_col)
{
    float result = 0.0f;
    #pragma unroll
    for (int n = 0; n < TILE_SIZE; ++n) {
        result += As[local_row * GROUP_SIZE_X + n] * Bs[n * GROUP_SIZE_X + local_col];
    }

    return result;
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
    const unsigned int row = get_group_id(1) * GROUP_SIZE_X + get_local_id(1);
    const unsigned int col = get_group_id(0) * GROUP_SIZE_X + get_local_id(0);

    const unsigned int local_row = get_local_id(1);
    const unsigned int local_col = get_local_id(0);

    __local float A_tiles[GROUP_SIZE_Y * GROUP_SIZE_X];
    __local float B_tiles[GROUP_SIZE_Y * GROUP_SIZE_X];

    float result = 0.0f;
    const unsigned int num_tiles = k / GROUP_SIZE_X;

    for (unsigned int t = 0; t < num_tiles; ++t) {
        load_tile_A(a, A_tiles, row, k, t, local_row, local_col);
        load_tile_B(b, B_tiles, col, w, t, local_row, local_col);

        barrier(CLK_LOCAL_MEM_FENCE);

        result += compute_tile_product(A_tiles, B_tiles, local_row, local_col);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row >= h || col >= w) return;
    c[row * w + col] = result;
}
