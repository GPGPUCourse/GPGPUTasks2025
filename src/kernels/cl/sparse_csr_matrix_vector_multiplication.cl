#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(__global const uint* csr_row_offsets,
                                                      __global const uint* csr_columns,
                                                      __global const uint* csr_values,
                                                      __global const uint* vector_values,
                                                      __global       uint* output_vector_values,
                                                             unsigned int  nrows)
{
    unsigned int row = get_global_id(0);

    if (row >= nrows)
        return;

    unsigned int row_begin = csr_row_offsets[row];
    unsigned int row_end = csr_row_offsets[row + 1];

    rassert(row_begin <= row_end, 471921);

    ulong accum= 0;
    for (unsigned int i = row_begin; i < row_end; ++i) {
        unsigned int column = csr_columns[i];
        unsigned int matrix_value = csr_values[i];
        unsigned int vector_value = vector_values[column];
        accum += (ulong)matrix_value * (ulong)vector_value;
    }

    output_vector_values[row] = (uint)accum;
}
