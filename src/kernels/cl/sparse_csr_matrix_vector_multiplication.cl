#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector_values,
    uint nrows,
    uint ncols)
{
    const uint row = get_global_id(0);

    if (row >= nrows) {
        return;
    }

    const uint row_begin = csr_row_offsets[row];
    const uint row_end = csr_row_offsets[row + 1];

    rassert(row_begin <= row_end, 435623452);
    rassert(row_end <= csr_row_offsets[nrows], 734562345);

    uint accumulator = 0;

    for (uint idx = row_begin; idx < row_end; ++idx) {
        const uint column = csr_columns[idx];
        rassert(column < ncols, 523452345);

        accumulator += csr_values[idx] * vector_values[column];
    }

    output_vector_values[row] = accumulator;
}
