#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
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
                   uint  nrows 
)
{
    uint row = get_global_id(0);
    if (row >= nrows) {
        return;
    }

    uint row_from = csr_row_offsets[row];
    uint row_to = csr_row_offsets[row + 1];

    ulong accumulator = 0;
    for (uint i = row_from; i < row_to; ++i) {
        uint col = csr_columns[i];
        accumulator += (ulong)csr_values[i] * (ulong)vector_values[col];
    }
    output_vector_values[row] = (uint)accumulator;
}
