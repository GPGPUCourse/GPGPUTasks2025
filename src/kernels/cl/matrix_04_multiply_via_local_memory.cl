#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_X, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float local_data_a[GROUP_SIZE];
    __local float local_data_b[GROUP_SIZE];

    // rassert(get_local_size(0) == get_local_size(1), 452345776);

    const uint local_size = get_local_size(0);

    const uint col = get_global_id(0);
    const uint row = get_global_id(1);

    const uint index_c = row * w + col;

    const uint local_col = get_local_id(0);
    const uint local_row = get_local_id(1);

    const uint local_index = local_row * local_size + local_col;

    const uint group_col = get_group_id(0);
    const uint group_row = get_group_id(1);

    float acc = 0.0;
    const uint group_count = (k + local_size - 1) / local_size;
    for (uint i = 0; i < group_count; ++i) {
        const uint col_a = i * local_size + local_col;
        const uint row_a = row;
        const uint index_a = row_a * k + col_a;

        if (col_a < k && row_a < h) {
            local_data_a[local_index] = a[index_a];
        } else {
            local_data_a[local_index] = 0.0;
        }

        const uint col_b = col;
        const uint row_b = i * get_local_size(1) + local_row;
        const uint index_b = row_b * w + col_b;

        if (col_b < w && row_b < k) {
            local_data_b[local_index] = b[index_b];
        } else {
            local_data_b[local_index] = 0.0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < local_size; ++j) {
            acc += local_data_a[local_row * local_size + j] * local_data_b[j * local_size + local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (col < w && row < h) {
        c[index_c] = acc;
    }
}
