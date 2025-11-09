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
    const uint nrows)
{
    const uint row = get_group_id(0);
    const uint local_index = get_local_id(0);
    const uint local_size = get_local_size(0);
    if (row >= nrows) {
        return;
    }

    const uint from = csr_row_offsets[row];
    const uint to = csr_row_offsets[row + 1];
    uint sum = 0;
    for (uint idx = from + local_index; idx < to; idx += local_size) {
        sum += csr_values[idx] * vector_values[csr_columns[idx]];
    }

    __local uint local_sum[GROUP_SIZE];
    local_sum[local_index] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint step = local_size >> 1; step > 0; step >>= 1) {
        if (local_index < step) {
            local_sum[local_index] += local_sum[local_index + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        output_vector_values[row] = local_sum[0];
    }
}
