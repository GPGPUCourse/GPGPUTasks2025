#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // ys=h x xs=k
                       __global const float* b, // ys=k x xs=w
                       __global       float* c, // ys=h x xs=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float local_a[GROUP_SIZE];
    __local float local_b[GROUP_SIZE];

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const uint l_w = get_local_size(0);

    const uint l_x = get_local_id(0);
    const uint l_y = get_local_id(1);

    if (y < h && x < w) {
        c[y * w + x] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < k; i += l_w)
    {
        const uint ind = l_y * l_w + l_x;
        
        if (y < h && x < w) {
            local_a[ind] = a[y * k + (i + l_x)];
            local_b[ind] = b[(i + l_y) * w + x];
        } else {
            local_a[ind] = 0;
            local_b[ind] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (y < h && x < w) {
            for (int j = 0; j < l_w; j++) {
                c[y * w + x] += local_a[l_y * l_w + j] * local_b[j * l_w + l_x];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
