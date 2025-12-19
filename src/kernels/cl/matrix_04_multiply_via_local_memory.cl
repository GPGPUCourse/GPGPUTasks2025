#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const uint col = get_global_id(0);
    const uint row = get_global_id(1);

    const uint col_loc = get_local_id(0);
    const uint row_loc = get_local_id(1);

    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    __local float a_tile[16][16];
    __local float b_tile[16][16];
    float acc = 0;

    for (int tile = 0; tile < (k+15)/16; tile++){

        int ay = group_y*16+row_loc, ax = tile*16+col_loc;
        a_tile[row_loc][col_loc] = (ay < h && ax < k) ? a[ay*k+ax] : 0;

        int bx = group_x*16+col_loc, by = tile*16+row_loc;
        b_tile[row_loc][col_loc] = (by < k && bx < w) ? b[by*w+bx] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);
        for(int t = 0; t<16; t++){
            acc += a_tile[row_loc][t] * b_tile[t][col_loc];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < h && col < w) {
        c[row * w + col] = acc;
    }
}
