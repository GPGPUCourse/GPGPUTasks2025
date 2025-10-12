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
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    uint up_x = get_group_id(0) * GROUP_SIZE_X;
    uint up_y = get_group_id(1) * GROUP_SIZE_X;

    uint lx = get_local_id(0);
    uint ly = get_local_id(1);

    __local float local_a[GROUP_SIZE_X][GROUP_SIZE_X];
    __local float local_b[GROUP_SIZE_X][GROUP_SIZE_X];

    float sum = 0.0f;
    for(uint up_z = 0; up_z < k; up_z += GROUP_SIZE_X) {

        local_a[ly][lx] = a[y * k + up_z + lx];
        local_b[ly][lx] = b[(up_z + ly) * w + (up_x + lx)];

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (uint lz = 0; lz < GROUP_SIZE_X; ++lz) {
            sum += local_a[ly][(lx+lz)%GROUP_SIZE_X] * local_b[(lx+lz)%GROUP_SIZE_X][lx];
        }
    
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[y * w + x] = sum;
}
