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
    unsigned int glob_x = get_global_id(0);
    unsigned int glob_y = get_global_id(1);
    unsigned int loc_x = get_local_id(0);
    unsigned int loc_y = get_local_id(1);
    __local float tilea[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tileb[GROUP_SIZE_Y][GROUP_SIZE_X];
    float s = 0.f;
    unsigned int tilen = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_Y;
    for (int i = 0; i < tilen; ++i) {
        int x = i * GROUP_SIZE_Y + loc_x;
        if (glob_y >= h && x >= k) {
            tilea[loc_y][loc_x] = 0.f;
        } else {
            tilea[loc_y][loc_x] = a[glob_y * k + x];
        }
        int y = i * GROUP_SIZE_X + loc_y;
        if (glob_x >= w && y >= k) {
            tileb[loc_y][loc_x] = 0.0f;
        } else {
            tileb[loc_y][loc_x] = b[y * w + glob_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int j = 0; j < GROUP_SIZE_X; ++j) {
            s += tilea[loc_y][j] * tileb[j][loc_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (glob_x < w && glob_y < h) {
        c[glob_y * w + glob_x] = s;
    }
}
