#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offset,
    __global const uint* col_indices,
    __global const uint* matrix_values,
    __global const uint* vector_values,
    __global uint* result,
    const int nrows
)
{
    const int row = get_group_id(0);
    if (row >= nrows) {
        return;
    }

    const uint local_id = get_local_id(0);
    const uint from = row_offset[row] + local_id;
    const uint to = row_offset[row + 1];

    local uint sums[GROUP_SIZE];

    uint sum = 0;

    for (uint i = from; i < to; i += GROUP_SIZE) {
        sum += matrix_values[i] * vector_values[col_indices[i]];
    }

    sums[local_id] = sum;

    for (uint shift = GROUP_SIZE / 2; shift > 0; shift >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < shift) {
            sums[local_id] += sums[local_id + shift];
        }
    }

    if (local_id == 0) {
        result[row] = sums[0];
    }
}