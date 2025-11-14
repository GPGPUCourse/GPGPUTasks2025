#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* input_vector,
    __global uint* output_vector,
    const uint rows_number)
{
    int row = get_global_id(0);

    if (row >= rows_number) {
        return;
    }

    uint sum = 0;
    for (uint i = row_offsets[row]; i < row_offsets[row + 1]; ++i) {
        uint col = columns[i];
        sum += values[i] * input_vector[col];
    }
    output_vector[row] = sum;
}
