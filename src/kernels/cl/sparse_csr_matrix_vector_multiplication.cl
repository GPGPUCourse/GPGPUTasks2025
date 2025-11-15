#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector_values,
    __global uint* result,
    uint nnz, uint nrows)
{
    const uint row = get_group_id(0);
    const uint offset = row_offsets[row];
    uint row_size = 0;
    if (row + 1 == nrows) {
        row_size = nnz - offset;
    } else {
        row_size = row_offsets[row + 1] - offset;
    }

    uint local_index = get_local_id(0);
    __local uint local_sum[GROUP_SIZE];
    local_sum[local_index] = 0;
    uint temp_sum = 0;

    for (uint index = local_index; index < row_size; index += GROUP_SIZE) {
        uint global_index = offset + index;
        uint column = columns[global_index];
        temp_sum += values[global_index] * vector_values[column];
    }

    local_sum[local_index] = temp_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        uint sum = 0;
        for (uint i = 0; i < GROUP_SIZE; ++i) {
            sum += local_sum[i];
        }
        result[row] = sum;
    }
}
