#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* cols, 
    __global const uint* values,
    __global const uint* vector_values,
    __global uint* output_vector,
    unsigned int rows_cnt)
{
    const unsigned int row_idx = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_size = get_local_size(0);

    if (row_idx >= rows_cnt) {
        return;
    }

    const unsigned int row_start = row_offsets[row_idx];
    const unsigned int row_end = row_offsets[row_idx + 1];

    unsigned int partial_sum = 0; 
    for (unsigned int i = row_start + local_id; i < row_end; i += group_size) {
        const unsigned int col_idx = cols[i];
        const unsigned int val = values[i];

        partial_sum += val * vector_values[col_idx];
    }
    

    __local unsigned int local_sums[GROUP_SIZE];

    local_sums[local_id] = partial_sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_sums[local_id] += local_sums[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        output_vector[row_idx] = local_sums[0];
    }
}
