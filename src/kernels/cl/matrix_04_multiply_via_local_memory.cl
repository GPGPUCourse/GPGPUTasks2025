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
    __local float part_a[16][16];
    __local float part_b[16][16];

    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    int parts_count = (k + 15) / 16;

    float sum = 0;

    for (int p = 0; p < parts_count; ++p) {
        int px = p * 16 + lx;
        int py = p * 16 + ly;

        // a: h(y) x k(px)
        if (y < h && px < k) {
            part_a[ly][lx] = a[y * k + px];
        } else {
            part_a[ly][lx] = 0;
        }

        // b: k(py) x w(x)
        if (x < w && py < k) {
            part_b[ly][lx] = b[py * w + x];
        } else {
            part_b[ly][lx] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < 16; ++i) {
            sum += part_a[ly][i] * part_b[i][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE); // important!
    }

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
