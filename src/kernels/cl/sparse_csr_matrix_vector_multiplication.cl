#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(WARP_SIZE, 8, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector,
    __global uint* result,
    uint nrows,
    uint ncols
)
{
    const uint t_id = get_local_id(0);
    const uint local_row = get_local_id(1);
    const uint row = get_global_id(1);
    __local uint acc[WARP_SIZE * 8];
    acc[local_row * WARP_SIZE + t_id] = 0;

    if (row >= nrows) return;

    const uint row_offset = csr_row_offsets[row];
    const uint row_values_count = csr_row_offsets[row + 1] - row_offset;
    const uint required_iters = (row_values_count + WARP_SIZE - 1) / WARP_SIZE; // round up

    for (uint k = 0; k < required_iters; ++k) {
        const uint my_offset = row_offset + k * WARP_SIZE + t_id;
        if (my_offset < csr_row_offsets[row + 1]) {
            const uint my_column = csr_columns[my_offset];
            const uint my_value = csr_values[my_offset];
            const uint vector_value = vector[my_column];
            acc[local_row * WARP_SIZE + t_id] += my_value * vector_value;
        }
    }

    uint final_sum = 0;
    if (t_id == 0) {
        for (uint i = 0; i < WARP_SIZE; i++) final_sum += acc[local_row * WARP_SIZE + i];
        result[row] = final_sum;
    }
}
