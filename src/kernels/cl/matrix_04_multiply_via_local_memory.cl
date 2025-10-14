#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

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
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    if (x >= w || y >= h)
        return;

    __local float cache_b[GROUP_SIZE_Y][GROUP_SIZE_X];
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    float result = 0;
    for (unsigned int i0 = 0; i0 < k; i0 += GROUP_SIZE_Y) {
        const unsigned int b_x = x;
        const unsigned int b_y = i0 + local_y;
        if (b_x < w && b_y < k) {
            cache_b[local_y][local_x] = b[b_y * w + b_x];
        } else {
            cache_b[local_y][local_x] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = i0; i < i0 + GROUP_SIZE_Y; ++i) {
            result += a[y * k + i] * cache_b[i - i0][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[y * w + x] = result;
}
