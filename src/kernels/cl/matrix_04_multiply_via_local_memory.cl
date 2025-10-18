#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

//works only with square work groups
__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_X, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float block_a[GROUP_SIZE_X][GROUP_SIZE_X];
    __local float block_b[GROUP_SIZE_X][GROUP_SIZE_X];
    
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    float result = 0;

    for (int i = 0; i < k; i += GROUP_SIZE_X) {
        if (y < h && i + lx < k) {
            block_a[ly][lx] = a[y * k + i + lx];
        }

        if (x < w && i + ly < k) {
            block_b[ly][lx] = b[(ly + i) * w + x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < GROUP_SIZE_X; j++) {
            result += block_a[ly][j] * block_b[j][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = result;
    }
}
