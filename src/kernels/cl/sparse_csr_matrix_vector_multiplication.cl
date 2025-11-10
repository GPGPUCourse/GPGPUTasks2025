#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* result_buffer,
    const uint row_cnt,
    const uint column_cnt
    )
{
    const uint result_index = get_global_id(0);
    if (result_index > row_cnt) {
        return;
    }
    uint offset_start = csr_offsets[result_index];
    uint end_index = column_cnt;
    if (result_index < row_cnt) {
        end_index = csr_offsets[result_index + 1];
    }
    int s = 0;
    for (int i = offset_start; i < end_index; ++i) {
        s += csr_values[i] * vector_values[csr_columns[i]];
    }
    result_buffer[result_index] = s;
}
