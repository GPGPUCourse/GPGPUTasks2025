#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    rassert(GROUP_SIZE_X == GROUP_SIZE_Y, 72345890)
    rassert(k % GROUP_SIZE_X == 0, 57482390)

    const unsigned int local_row = get_local_id(1);
    const unsigned int local_col = get_local_id(0);

    const unsigned int global_row = get_global_id(1);
    const unsigned int global_col = get_global_id(0);

    __local float tileA[GROUP_SIZE_X][GROUP_SIZE_Y];
    __local float tileB[GROUP_SIZE_X][GROUP_SIZE_Y];
    const unsigned int num_tiles = k / GROUP_SIZE_X;

    float sum = 0.0f;
    for (unsigned int i = 0; i < num_tiles; ++i) {

        const unsigned int tiled_col = i * GROUP_SIZE_X + local_col;
        const unsigned int tiled_row = i * GROUP_SIZE_Y + local_row;

        tileA[local_row][local_col] = a[global_row * k + tiled_col];
        tileB[local_row][local_col] = b[tiled_row * w + global_col];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < GROUP_SIZE_X; ++j) {
            sum += tileA[local_row][j] * tileB[j][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_row * w + global_col] = sum;
}
