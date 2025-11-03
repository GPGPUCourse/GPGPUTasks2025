#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

//__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const unsigned int* csr_row_offsets,
    __global const unsigned int* csr_columns,
    __global const unsigned int* csr_values,
    __global const unsigned int* vector_values,
    __global       unsigned int* result_vector_values,
                   unsigned int  nrows,
                   unsigned int  ncols
) {
    const unsigned int global_row = get_global_id(0);
    const unsigned int local_offset = get_local_id(1);
    const unsigned int local_row = get_local_id(0);

    __local unsigned int local_accumulator[GROUP_SIZE_Y * GROUP_SIZE_X];
    if (global_row < nrows) {
        unsigned int accumulator = 0;
        const unsigned int row_from = csr_row_offsets[global_row];
        const unsigned int row_to = csr_row_offsets[global_row + 1];

        for (unsigned int i = row_from + local_offset; i < row_to; i += GROUP_SIZE_Y) {
            const unsigned int col = csr_columns[i];
            accumulator += csr_values[i] * vector_values[col];
        }

        local_accumulator[local_row * GROUP_SIZE_Y + local_offset] = accumulator;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_row < nrows && local_offset == 0) {
        unsigned int accumulator = 0;
        __local unsigned int* data = local_accumulator + local_row * GROUP_SIZE_Y;

        for (unsigned int i = 0; i < GROUP_SIZE_Y; ++i) {
            accumulator += data[i];
        }

        result_vector_values[global_row] = accumulator;
    }
}
