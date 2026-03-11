#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* input_vector,
    __global uint* output_vector,
    unsigned int  nrows,
    unsigned int  ncols,
    unsigned int  nnz)
{
    const uint index = get_local_id(0);
    const uint row = get_group_id(0);

    __local uint tile[GROUP_SIZE];

    const uint global_idx = row_offsets[row] + index;
    const uint next_offset = (row + 1 < nrows) ? row_offsets[row + 1] : nnz;

    const uint col = columns[global_idx];
    tile[index] = global_idx < next_offset ? values[global_idx] * input_vector[col] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (index != 0)
        return;

    unsigned int sum = 0;
    for(unsigned int t = 0; t < GROUP_SIZE; ++t) {
        sum += tile[t];
    }

    output_vector[row] = sum;
}
