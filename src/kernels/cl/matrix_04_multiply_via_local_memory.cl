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
    const unsigned int global_x = get_global_id(0); 
    const unsigned int global_y = get_global_id(1);
    const unsigned int local_x = get_local_id(0); 
    const unsigned int local_y = get_local_id(1);

    const unsigned int tile_size = GROUP_SIZE_X;
    __local float local_a_data[tile_size][tile_size];
    __local float local_b_data[tile_size][tile_size];

    float acc = 0;
    for (int t = 0; t < k/tile_size; t++) {
        local_a_data[local_y][local_x] = a[global_y * k + (t * tile_size + local_x)];
        local_b_data[local_y][local_x] = b[(t * tile_size + local_y) * w + global_x];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < tile_size; i++) {
            acc += local_a_data[local_y][i] * local_b_data[i][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[global_y * w + global_x] = acc;
}
