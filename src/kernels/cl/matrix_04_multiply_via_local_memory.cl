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
    __local float a_tile[16][17];
    __local float b_tile[16][17];
    
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);
    
    const unsigned int global_x = group_x * 16 + local_x;
    const unsigned int global_y = group_y * 16 + local_y;
    
    float sum = 0.0f;
    
    for (unsigned int tile = 0; tile < (k + 15) / 16; ++tile) {
        unsigned int a_col = tile * 16 + local_x;
        unsigned int b_row = tile * 16 + local_y;
        
        if (global_y < h && a_col < k) {
            a_tile[local_y][local_x] = a[global_y * k + a_col];
        }
        if (b_row < k && global_x < w) {
            b_tile[local_y][local_x] = b[b_row * w + global_x];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (unsigned int i = 0; i < 16; ++i) {
            sum += a_tile[local_y][i] * b_tile[i][local_x];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_x < w && global_y < h) {
        c[global_y * w + global_x] = sum;
    }
}
