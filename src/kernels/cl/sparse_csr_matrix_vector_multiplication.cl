#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global       uint* output_vector_values,
    unsigned int nrows)
{
    uint row = get_global_id(0);
    if (row >= nrows) {
        return;
    }

    uint sum = 0;
    uint start = csr_row_offsets[row];
    uint end = csr_row_offsets[row + 1];

    for (uint i = start; i < end; ++i) {
        sum += csr_values[i] * vector_values[csr_columns[i]];
    }
    output_vector_values[row] = sum;
}