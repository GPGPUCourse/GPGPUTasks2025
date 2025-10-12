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
    const uint global_index_c_x = get_global_id(0);
    const uint global_index_c_y = get_global_id(1);

    const uint local_index_x = get_local_id(0);
    const uint local_index_y = get_local_id(1);
    const uint local_index = local_index_y * 16 + local_index_x;

    __local float local_copy_a[16 * 16];
    __local float local_copy_b[16 * 16];

    float accumulated = 0;

    for (uint i = 0; i < k; i += 16) {
        uint global_index_a_x = i + local_index_x;
        local_copy_a[local_index] = 0;
        local_copy_b[local_index] = 0;

        if (global_index_c_y < h && global_index_a_x < k) {
            local_copy_a[local_index] = a[global_index_c_y * k + global_index_a_x];
        }

        uint global_index_b_y = i + local_index_y;
        if (global_index_c_x < w && global_index_b_y < k) {
            local_copy_b[local_index] = b[global_index_b_y * w + global_index_c_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint common = 0; common < 16; ++common) {
            accumulated += local_copy_a[local_index_y * 16 + common] * local_copy_b[common * 16 + local_index_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_index_c_y < h && global_index_c_x < w) {
        c[global_index_c_y * w + global_index_c_x] = accumulated;
    }
}
