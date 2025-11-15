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
    __global const uint* vector_values,
    __global       uint* output_vector_values,
             const uint  nrows)
 {
    const uint row = get_group_id(0);
    if (row >= nrows) {
        return;
    }
    const uint local_size = get_local_size(0);
    const uint local_id = get_local_id(0);

    const uint row_start = row_offsets[row];
    const uint row_end = row_offsets[row + 1] - 1;
    uint sum = 0;
    for (uint i = row_start + local_id; i <= row_end; i += local_size) {
        sum += values[i] * vector_values[columns[i]];
    }

    __local uint local_sums[GROUP_SIZE];
    local_sums[local_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint reduction_step = local_size >> 1; reduction_step != 0; reduction_step >>= 1) {
        if (local_id < reduction_step) {
            local_sums[local_id] += local_sums[local_id + reduction_step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        output_vector_values[row] = local_sums[0];
    }
}
