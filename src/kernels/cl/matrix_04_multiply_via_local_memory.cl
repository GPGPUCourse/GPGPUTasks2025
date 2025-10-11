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
    __local float data_a[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    __local float data_b[GROUP_SIZE_X][GROUP_SIZE_Y + 1];
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);


    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    if (x >= w || y >= h) {
        return;
    }
    float sum = 0.f;
    int br = 0;
    for (int i = 0; i < k; i++) {
        unsigned int idx_x = local_x + i * GROUP_SIZE_X;
        unsigned int idx_y = local_y + i * GROUP_SIZE_Y;
        if (idx_x < k) {
            data_a[local_y][local_x] = a[y * k + idx_x];
        } else {
            br |= 1;
            data_a[local_y][local_x] = .0f;
        }
        if (idx_y < k) {
            data_b[local_y][local_x] = b[idx_y * w + x];
        } else {
            br |= 2;
            data_b[local_y][local_x] = .0f;
        }
        if (br == 3) {
            break;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < GROUP_SIZE_X; ++j) {
            sum += data_a[local_y][j] * data_b[j][local_x];
        }
    }
}
