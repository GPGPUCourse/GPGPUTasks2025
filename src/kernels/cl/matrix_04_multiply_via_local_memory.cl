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
    __local float As[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    __local float Bs[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    float acc = 0;

    for (int t = 0; t * GROUP_SIZE_X < k; t++) {
        As[ly][lx] = (y < h && t * GROUP_SIZE_X + lx < k) ? a[y * k + t * GROUP_SIZE_X + lx] : 0.0f;
        Bs[ly][lx] = (t * GROUP_SIZE_Y + ly < k && x < w) ? b[(t * GROUP_SIZE_Y + ly) * w + x] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < GROUP_SIZE_X; i++)
            acc += As[ly][i] * Bs[i][lx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h)
        c[y * w + x] = acc;
}
