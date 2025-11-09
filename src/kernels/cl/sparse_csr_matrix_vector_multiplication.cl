#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector_values,
    const uint nrows) // DONE input/output buffers
{
    // DONE

    const uint global_id = get_global_id(0);

    if (global_id >= nrows) {
        return;
    }

    uint sum = 0;
    for (uint i = csr_row_offsets[global_id]; i < csr_row_offsets[global_id + 1]; ++i) {
        sum += csr_values[i] * vector_values[csr_columns[i]];
    }

    output_vector_values[global_id] = sum;
}
