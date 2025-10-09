#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"
#pragma OPENCL EXTENSION cl_khr_fp16 : enable // для half'ов

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                       const unsigned int w,
                       const unsigned int h,
                       const unsigned int k)
{
    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);
    if (row >= h || col >= w) return;

    const int local_col = get_local_id(0);
    const int local_row = get_local_id(1);

    __local half local_data_a[GROUP_SIZE_Y * (GROUP_SIZE_X + 1)];
    __local half local_data_b[GROUP_SIZE_Y * (GROUP_SIZE_X + 1)];
    // перемножение half'ов будет не сильно хуже перемножения float'ов, учитывая что результат в любом случае должен быть float
    // добавил лишний столбец чтобы избегать banl-конфликтов (но мог как и во 2 задании просто биекцию использовать)

    float acc = 0.0f;

    for (unsigned int it = 0; it < k; it += 16) {
        local_data_a[local_row * (GROUP_SIZE_X + 1) + local_col] = (it + local_col < k) ? a[row * k + it + local_col] : 0.0h;
        local_data_b[local_row * (GROUP_SIZE_X + 1) + local_col] = (it + local_row < k) ? b[(it + local_row) * w + col] : 0.0h;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < 16; ++j) {
            acc += (float)local_data_a[local_row * (GROUP_SIZE_X + 1) + j] * (float)local_data_b[j * (GROUP_SIZE_X + 1) + local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[row * w + col] = acc;
}
