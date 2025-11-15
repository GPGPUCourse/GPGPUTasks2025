#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
        __global const uint* csr_row_offset,
        __global const uint* csr_columns,
        __global const uint* csr_values,
        __global const uint* vector_values,
        __global uint* output_values,
        const uint nrows)
{
    const uint idx = get_global_id(0);

    if (idx >= nrows)
        return;
    
    uint row_begin = csr_row_offset[idx];
    uint row_end = csr_row_offset[idx + 1];

    ulong res = 0;
    for (int i = row_begin; i < row_end; ++i) {
        uint column = csr_columns[i];
        uint matrix_val = csr_values[i];
        uint vector_val = vector_values[column];
        res += matrix_val * vector_val;
    }

    output_values[idx] = res;
}
