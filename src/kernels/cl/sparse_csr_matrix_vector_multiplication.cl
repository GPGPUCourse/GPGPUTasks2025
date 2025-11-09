#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint*    csr_row_offsets,
    __global const uint*    csr_columns,
    __global const uint*    csr_values,
    __global const uint*    vector_values,
    __global       uint*    output_vector_values,
    const uint              nrows
)
{
    const uint row = get_global_id(0);

    if (row >= nrows) return;

    ulong ac = 0;
    for (uint i = csr_row_offsets[row]; i < csr_row_offsets[row+ 1]; ++i) {
        ac += (ulong)csr_values[i] * (ulong)vector_values[csr_columns[i]];
    }
    output_vector_values[row] = (uint)ac;
}
