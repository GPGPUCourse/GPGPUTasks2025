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
    __global uint*       output_values,
                   uint  nrows)
{
    const uint row = get_global_id(0);

    if (row >= nrows)
        return;

    const uint row_start = csr_row_offsets[row];
    const uint row_end = csr_row_offsets[row + 1];

    ulong sum = 0;
    for (uint idx = row_start; idx < row_end; ++idx) {
        const uint col = csr_columns[idx];
        sum += (ulong)csr_values[idx] * (ulong)vector_values[col];
    }

    output_values[row] = (uint)sum;
}
