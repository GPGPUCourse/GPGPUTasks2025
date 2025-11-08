#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets, // size: nrows + 1
    __global const uint* csr_columns,     // size: nnz
    __global const uint* csr_values,      // size: nnz
    __global const uint* vector_values,   // size: ncols
    __global       uint* output,          // size: nrows
             const uint nrows
)
{
    const uint row = get_global_id(0);
    if (row >= nrows) return;

    const uint row_from = csr_row_offsets[row];
    const uint row_to = csr_row_offsets[row + 1];

    ulong sum = 0ul;

    for (uint i = row_from; i < row_to; ++i)
        sum += (ulong)csr_values[i] * (ulong)vector_values[csr_columns[i]];

    output[row] = (uint)sum;
}
