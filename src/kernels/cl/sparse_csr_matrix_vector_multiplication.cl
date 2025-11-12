#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector_values,
    const uint nrows,
    const uint ncols
) {
    __local uint local_sums[GROUP_SIZE_X * GROUP_SIZE_Y];
    uint i = get_global_id(1);
    uint loc_i = get_local_id(1);
    uint local_index = get_local_id(0);
    uint row_start = csr_row_offsets[i];
    uint row_end = csr_row_offsets[i + 1];
    local_sums[loc_i * GROUP_SIZE_X + local_index] = 0;
    for (uint j = row_start + local_index; j < row_end; j += GROUP_SIZE_X) {
        uint col = csr_columns[j];
        uint val = csr_values[j];
        local_sums[loc_i * GROUP_SIZE_X + local_index] += val * vector_values[col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        uint sum = 0;
        for (uint j = 0; j < GROUP_SIZE_X; j++) {
            sum += local_sums[loc_i * GROUP_SIZE_X + j];
        }
        output_vector_values[i] = sum;
    }
}
