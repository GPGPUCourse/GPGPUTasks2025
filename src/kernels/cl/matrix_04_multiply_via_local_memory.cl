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
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint lx = get_local_id(0);
    uint ly = get_local_id(1);

    if (x >= w || y >= h) {
        return;
    }
    __local float a_local[16][16];
    __local float b_local[16][16];
    float sum = 0;
    for (int i = 0; i < k / 16; ++i) {
        a_local[ly][lx] = a[y * k + i * 16 + lx];
        b_local[ly][lx] = b[(i * 16 + ly) * w + x];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < 16; ++j) {
            sum += a_local[ly][j] * b_local[j][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[y * w + x] = sum;
}
