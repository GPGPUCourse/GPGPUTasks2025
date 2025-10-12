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
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);
    
    local float local_a[GROUP_SIZE_Y * GROUP_SIZE_X];
    local float local_b[GROUP_SIZE_Y * GROUP_SIZE_X];

    float accumulator = 0.0f;

    for (unsigned int k_block = 0; k_block < k; k_block += GROUP_SIZE_X) {
        const unsigned int a_row = group_y * GROUP_SIZE_Y + local_y;
        const unsigned int a_col = k_block + local_x;
        const unsigned int local_a_idx = local_y * GROUP_SIZE_X + local_x;
        if (a_row < h && a_col < k) {
            local_a[local_a_idx] = a[a_row * k + a_col];
        } else {
            local_a[local_a_idx] = 0.0f;
        }
        
        const unsigned int b_row = k_block + local_y;
        const unsigned int b_col = group_x * GROUP_SIZE_X + local_x;
        const unsigned int local_b_idx = local_y * GROUP_SIZE_X + local_x;
        if (b_row < k && b_col < w) {
            local_b[local_b_idx] = b[b_row * w + b_col];
        } else {
            local_b[local_b_idx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < GROUP_SIZE_X; i++) {
            const unsigned int a_idx = local_y * GROUP_SIZE_X + i;
            const unsigned int b_idx = i * GROUP_SIZE_X + local_x;
            accumulator += local_a[a_idx] * local_b[b_idx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = accumulator;
    }
}
