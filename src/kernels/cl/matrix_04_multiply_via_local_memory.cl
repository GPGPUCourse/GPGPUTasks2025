#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    int glob_x = get_group_id(0);
    int glob_y = get_group_id(1);
    int loc_x = get_local_id(0);
    int loc_y = get_local_id(1);

    __local float part_a[16][16];
    __local float part_b[16][16];

    float sum = 0.0f;

    int parts = (k + 16 - 1) / 16;
    for (int part_id = 0; part_id < parts; ++part_id) {
        int a_shift = part_id * 16 + loc_x;
        int b_shift = part_id * 16 + loc_y;
        if (glob_y < h && a_shift < k) {
            part_a[loc_y][loc_x] = a[a_shift + glob_y * k];
        } else {
            part_a[loc_y][loc_x] = 0.0f;
        }
        if (b_shift < k && glob_x < w) {
            part_b[loc_y][loc_x] = b[b_shift * w + glob_x];
        } else {
            part_b[loc_y][loc_x] = 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < 16; ++i) {
            sum += part_a[loc_y][i] * part_b[i][loc_x];
        }
    }
    if (glob_y < h && glob_x < w) {
        c[glob_y * w + glob_x] = sum;
    }
}
