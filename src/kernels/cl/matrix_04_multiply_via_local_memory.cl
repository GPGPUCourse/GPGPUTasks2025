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
    int x = get_global_offset(1); // 0..h
    int y = get_global_offset(0); // 0..w
    int i = get_local_id(1);
    int j = get_local_id(0);
    __local float a_tile[16][17];
    __local float b_tile[16][17];
    __local float c_tile[16][17];

    for (int z = 0; z < k; z += 16) {
        if (z + j < k) {
            a_tile[i][j] = a[(x + i) * k + (z + j)]; // sliding along 0..k - 2nd dim
        } else {
            a_tile[i][j] = 0;
        }
        if (z + i < k) {
            b_tile[i][j] = b[(z + i) * w + (y + j)]; // sliding along 0..k - 1st dim
        } else {
            b_tile[i][j] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // try to calculate with diagonal pattern to eliminate bank conflicts
        // gone bad locally :(
        int pi = j;
        int pj = (i + j) % 16;
        for (int t = 0; t < 16; ++t) {
            c_tile[pi][pj] += a_tile[pi][t] * b_tile[t][pj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[(x + i) * w + (y + j)] = c_tile[i][j];
}
