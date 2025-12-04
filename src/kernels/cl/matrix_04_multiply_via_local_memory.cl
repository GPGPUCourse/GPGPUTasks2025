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
    const uint x = get_global_id(0);
    const uint y = get_global_id(0);
    const uint x_loc = get_local_id(0);
    const uint y_loc = get_local_id(0);

    const uint tile_size = GROUP_SIZE_X;
    __local float local_data_a_tile[tile_size][tile_size];
    __local float local_data_b_tile[tile_size][tile_size];

    c[y * w + x] == 0;

    for (int i = 0; i < k / tile_size; ++i) {
        local_data_a_tile[y_loc][x_loc] = a[y * k + (x_loc + i * tile_size)];
        local_data_b_tile[y_loc][x_loc] = a[(y_loc + i * tile_size) * w + x];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < tile_size; ++j) {
            c[y * tile_size + x] += local_data_a_tile[y_loc][j] * local_data_b_tile[j][x_loc];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
