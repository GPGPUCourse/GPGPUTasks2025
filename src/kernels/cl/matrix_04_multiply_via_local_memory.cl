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
    __local float local_data_a[GROUP_SIZE_X * GROUP_SIZE_Y];
    __local float local_data_b[GROUP_SIZE_X * GROUP_SIZE_Y];

    const uint i = get_global_id(0);
    const uint j = get_global_id(1);
    const uint local_i = get_local_id(0);
    const uint local_j = get_local_id(1);

    const uint group_count = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;
    float acc = 0.0f;

    for (int s = 0; s < group_count; ++s) {
        // Чтение
        const uint a_i = s * GROUP_SIZE_X + local_i;
        const uint a_j = j;

        if (a_i < k && a_j < h) {
            local_data_a[local_i + local_j * GROUP_SIZE_X] = a[a_i + a_j * k];
        } else {
            local_data_a[local_i + local_j * GROUP_SIZE_X] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const uint b_i = i;
        const uint b_j = s * GROUP_SIZE_X + local_j;
        
        if (b_i < w && b_j < k) {
            local_data_b[local_i + local_j * GROUP_SIZE_X] = b[b_i + b_j * w];
        } else {
            local_data_b[local_i + local_j * GROUP_SIZE_X] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Умножение внутри local_data
        for (int local_ki = 0; local_ki < GROUP_SIZE_X; ++local_ki) {
            acc += local_data_a[local_j * GROUP_SIZE_X + local_ki] * 
                        local_data_b[local_ki * GROUP_SIZE_X + local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Запись 
    c[j * w + i] = acc;
}