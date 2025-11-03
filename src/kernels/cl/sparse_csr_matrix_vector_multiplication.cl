#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const unsigned int* csr_row_offsets,
    __global const unsigned int* csr_columns,
    __global const unsigned int* csr_values,
    __global const unsigned int* vector_values,
    __global       unsigned int* result_vector_values,
                   unsigned int  nrows,
                   unsigned int  ncols
) {
    const unsigned int row = get_global_id(0);

    if (row >= nrows) {
        return;
    }

    unsigned int accumulator = 0;
    const unsigned int row_from = csr_row_offsets[row];
    const unsigned int row_to = csr_row_offsets[row + 1];

    for (unsigned int i = row_from; i < row_to; ++i) {
        const unsigned int col = csr_columns[i];
        accumulator += csr_values[i] * vector_values[col];
    }

    result_vector_values[row] = accumulator;
}
