#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define BANK_SIZE 32

#define OFFSET (BANK_SIZE / GROUP_SIZE_X)
#define LOCAL_A_SIZE (GROUP_SIZE_X + OFFSET)

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
    const unsigned int tiles_count = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;
    const unsigned int shuffle_offset = OFFSET * (local_y / OFFSET);
    __local float local_a[LOCAL_A_SIZE * GROUP_SIZE_Y];
    __local float local_b[GROUP_SIZE_X * GROUP_SIZE_Y];
    float sum = 0;

    for (unsigned int tile_idx = 0; tile_idx < tiles_count; ++tile_idx) {
        unsigned int block_start = tile_idx * GROUP_SIZE_X;
        local_a[local_y * LOCAL_A_SIZE + (local_x + shuffle_offset) % GROUP_SIZE_X] = a[global_y * k + block_start + local_x];
        local_b[local_y * GROUP_SIZE_X + local_x] = b[(block_start + local_y) * w + global_x];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int z = 0; z < GROUP_SIZE_X; ++z) {
            sum += local_a[local_y * LOCAL_A_SIZE + (z + shuffle_offset) % GROUP_SIZE_X] * local_b[z * GROUP_SIZE_X + local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[global_y * w + global_x] = sum;
}
