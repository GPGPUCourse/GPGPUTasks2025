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
    __global uint* output_vector_values,
    unsigned int nrows)
{
    const uint row = get_group_id(0);
    const uint lid = get_local_id(0);

    if (row >= nrows)
        return;

    const uint row_from = csr_row_offsets[row];
    const uint row_to = csr_row_offsets[row + 1];

    ulong sum = 0;
    for (uint i = row_from + lid; i < row_to; i += GROUP_SIZE) {
        const uint col = csr_columns[i];
        sum += (ulong)csr_values[i] * (ulong)vector_values[col];
    }

    __local ulong partial[GROUP_SIZE];
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        ulong total = 0;
        for (uint i = 0; i < GROUP_SIZE; ++i) {
            total += partial[i];
        }
        output_vector_values[row] = (uint)total;
    }
}
