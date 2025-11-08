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
    const uint rows_size,
    const uint values_size,
    __global uint* output_vector_values
) // TODO input/output buffers
{
   const unsigned int idx = get_global_id(0);
    if (idx >= rows_size) {
        return;
    }

    const unsigned int offset = csr_row_offsets[idx];
    const unsigned int next_row_offset = idx == rows_size ? values_size : csr_row_offsets[idx + 1];
    unsigned int value = 0;

    for (unsigned int i = offset; i < next_row_offset; ++i) {
        value += csr_values[i] * vector_values[csr_columns[i]];
    }
    output_vector_values[idx] = value;
}
