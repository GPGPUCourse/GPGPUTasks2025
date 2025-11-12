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
    const uint nrows,
    const uint ncols
) {
    uint index = get_global_id(0);
    if (index >= nrows) {
        return;
    }
    uint row_start = csr_row_offsets[index];
    uint row_end;
    if (index + 1 < nrows) {
        row_end = csr_row_offsets[index + 1];
    } else {
        row_end = csr_row_offsets[nrows];
    }
    uint sum = 0;
    for (uint j = row_start; j < row_end; j++) {
        uint col = csr_columns[j];
        uint val = csr_values[j];
        sum += val * vector_values[col];
    }
    output_vector_values[index] = sum;
}
