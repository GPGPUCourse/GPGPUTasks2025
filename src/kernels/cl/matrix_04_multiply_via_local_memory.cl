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
    __local float local_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float local_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint lx = get_local_id(0);
    uint ly = get_local_id(1);

    float sum = 0.0f;
    for (uint t = 0; t < k; t += GROUP_SIZE_X) {
        uint tiled_k_a = t + lx;
        local_a[ly][lx] = a[y * k + tiled_k_a];

        uint tiled_k_b = t + ly;
        local_b[ly][lx] = b[tiled_k_b * w + x];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint kk = 0; kk < GROUP_SIZE_X; ++kk) { // assumption: GROUP_SIZE_X == GROUP_SIZE_Y
            sum += local_a[ly][kk] * local_b[kk][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[y * w + x] = sum;
}
