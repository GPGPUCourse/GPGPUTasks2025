#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const unsigned int* row_offsets, // csr_row_offsets
    __global const unsigned int* cols, // csr_columns
    __global const unsigned int* vals, // csr_values
    __global const unsigned int* x, // входной вектор
    __global unsigned int* y,  // выходной вектор
    const uint nrows)
{
    const int row = get_group_id(0);

    if (row >= nrows) {
        return;
    };

    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);

    __local unsigned int local_sum[256];
    local_sum[local_id] = 0;

    int row_start = row_offsets[row];
    int row_end = row_offsets[row + 1];

    for (int i = row_start + local_id; i < row_end; i += local_size) {
        local_sum[local_id] += vals[i] * x[cols[i]];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        unsigned int result = 0;
        for (unsigned int i = 0; i < GROUP_SIZE; i++) {
            result += local_sum[i];
        }
        y[row] = result;
    }
}
