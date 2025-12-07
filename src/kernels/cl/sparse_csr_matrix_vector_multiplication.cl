#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector,
    uint nrows)
{
    int row_id = get_group_id(0);
    int local_offset = get_local_id(0);

    __local int group_sums[GROUP_SIZE];

    if (row_id >= nrows) {
        return;
    }

    int start_index = csr_row_offsets[row_id];
    int end_index = csr_row_offsets[row_id + 1];

    if (local_offset + start_index >= end_index) {
        group_sums[local_offset] = 0;
    } else {
        group_sums[local_offset] = csr_values[local_offset + start_index] * vector_values[csr_columns[local_offset + start_index]];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_offset == 0) {
        uint sum = 0;
        for (uint i = 0; i < GROUP_SIZE; ++i) {
            sum += group_sums[i];
        }

        output_vector[row_id] = sum;
    }
}
