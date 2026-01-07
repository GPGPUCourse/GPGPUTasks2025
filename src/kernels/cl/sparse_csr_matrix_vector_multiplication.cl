#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* vector_values,
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global       uint* output,
    uint nrows,
    uint ncols,
    uint nnz
)
{
    const uint i = get_global_id(0);
    const uint local_i = get_local_id(0);
    const uint row = i / GROUP_SIZE;
    const uint row_offset = csr_row_offsets[row];
    const uint row_size = (row < nrows - 1) ? (csr_row_offsets[row + 1] - row_offset) : nnz - row_offset;

    __local uint res;

    if (local_i == 0) {
        res = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint in_row_index = i % GROUP_SIZE; in_row_index < row_size; in_row_index += GROUP_SIZE) {
        uint col = csr_columns[row_offset + in_row_index];
        uint mat_val = csr_values[row_offset + in_row_index];
        uint vec_val = vector_values[col];
        uint prod = vec_val * mat_val;
        atomic_add(&res, prod);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    output[row] = res;
}
