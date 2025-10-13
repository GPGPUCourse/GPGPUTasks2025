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
    // TODO
    __local float buff1[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float buff2[GROUP_SIZE_Y][GROUP_SIZE_X];
    
    const uint global_i = get_global_id(0);
    const uint global_j = get_global_id(1);
    
    const uint local_i = get_local_id(0);
    const uint local_j = get_local_id(1);
    
    float sum = 0.0f;
    
    const uint n = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;
    
    for (uint tile = 0; tile < n; ++tile) {
        const uint col1 = tile * GROUP_SIZE_X + local_i;
        const uint row1 = global_j;
        if (row1 < h && col1 < k) {
            buff1[local_j][local_i] = a[row1 * k + col1];
        } else {
            buff1[local_j][local_i] = 0.0f;
        }
        
        const uint col2 = global_i;
        const uint row2 = tile * GROUP_SIZE_X + local_j;
        if (row2 < k && col2 < w) {
            buff2[local_j][local_i] = b[row2 * w + col2];
        } else {
            buff2[local_j][local_i] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (uint ki = 0; ki < GROUP_SIZE_X; ++ki) {
            sum += buff1[local_j][ki] * buff2[ki][local_i];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_i < w && global_j < h) {
        c[global_j * w + global_i] = sum;
    }
}
