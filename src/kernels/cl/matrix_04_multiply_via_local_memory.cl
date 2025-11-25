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
    const int s = 16;
    __local float buffer1[s][s];
    __local float buffer2[s][s];
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    
    float res = 0.0f;
    
    for (int i = 0; i < (k + s - 1) / s; i++) {
        if (y < h && i * s + local_x < k) {
            buffer1[local_y][local_x] = a[y * k + i * s + local_x];
        } else {
            buffer1[local_y][local_x] = 0;
        }
        
        if (i * s + local_y < k && x < w) {
            buffer2[local_y][local_x] = b[(i * s + local_y) * w + x];
        } else {
            buffer2[local_y][local_x] = 0;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int i = 0; i < s; i++) {
            res += buffer1[local_y][i] * buffer2[i][local_x];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (x < w && y < h) {
        c[y * w + x] = res;
    }
}
