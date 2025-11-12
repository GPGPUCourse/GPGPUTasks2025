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
    const uint nrows,
    const uint ncols
) {
    __local uint local_sums[GROUP_SIZE];
    uint index = get_group_id(0);
    uint local_index = get_local_id(0);
    uint row_start = csr_row_offsets[index];
    uint row_end = csr_row_offsets[index + 1];
    local_sums[local_index] = 0;
    for (uint j = row_start + local_index; j < row_end; j += GROUP_SIZE) {
        uint col = csr_columns[j];
        uint val = csr_values[j];
        local_sums[local_index] += val * vector_values[col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        uint sum = 0;
        for (uint i = 0; i < GROUP_SIZE; i++) {
            sum += local_sums[i];
        }
        output_vector_values[index] = sum;
    }
}
