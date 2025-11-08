#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__kernel __attribute__((reqd_work_group_size(GROUP_SIZE_X, 1, 1)))
void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* x,
    __global uint* y,
    const uint nrows)
{
    const uint index = get_global_id(0);

    if (index >= nrows) {
        return;
    }

    const uint start = row_offsets[index];
    const uint end = row_offsets[index + 1];

    uint sum = 0;
    for (uint j = start; j < end; ++j) {
        sum += values[j] * x[columns[j]];
    }
    y[index] = sum;
}
