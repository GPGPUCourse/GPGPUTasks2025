#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint *row_offsets,
    __global const uint *col,
    __global const uint *values,
    __global const uint *x,
    __global uint *ans,
    const uint rows,
    const uint nnz)
{
    const uint row = get_group_id(0);
    if (row >= rows)
    {
        return;
    }
    const uint local_index = get_local_id(0);
    // printf("row: %d, index: %d\n", row, local_index);
    __local uint sum[GROUP_SIZE];
    sum[local_index] = 0;
    const uint row_begin = row_offsets[row];
    const uint row_end = row + 1 != rows ? row_offsets[row + 1] : nnz;
    for (uint i = row_begin + local_index; i < row_end; i += GROUP_SIZE)
    {
        sum[local_index] += values[i] * x[col[i]];
    }
    atomic_add(ans + row, sum[local_index]);
}
