#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_offset,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector,
    __global       uint* output,
    unsigned int nrows,
    unsigned int nnz)
{
    unsigned int row = get_group_id(0);
    unsigned int col = get_local_id(0);

    __local unsigned int local_sum[GROUP_SIZE];
    local_sum[col] = 0;

    if (row < nrows) {
        unsigned int frontier = row < nrows - 1 ? csr_offset[row + 1] : nnz;
        for (unsigned int my_idx = csr_offset[row] + col; my_idx < frontier; my_idx += GROUP_SIZE) {
            local_sum[col] += csr_values[my_idx] * vector[csr_columns[my_idx]];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // суммирование будет выполнять только поток 0.
    if (col == 0 && row < nrows) {
        unsigned int anw = 0;
        for (unsigned int i = 0; i < GROUP_SIZE; ++i) {
            anw += local_sum[i];
        }

        output[row] = anw;
    }
}
