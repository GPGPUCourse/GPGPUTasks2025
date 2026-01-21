#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global       uint* output_vector,
    const unsigned int nrows)
{
    const unsigned int row = get_global_id(0);

    if (row >= nrows) return;

    unsigned int row_start = csr_row_offsets[row];
    unsigned int row_end = csr_row_offsets[row + 1];

    size_t accumulator = 0;
    for (unsigned int i = row_start; i < row_end; i++) {
        unsigned int col = csr_columns[i];
        accumulator += csr_values[i] * vector_values[col];
    }

    output_vector[row] = accumulator;
}
