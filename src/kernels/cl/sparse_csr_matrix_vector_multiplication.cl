#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* restrict csr_row_offsets,
    __global const uint* restrict csr_columns,
    __global const uint* restrict csr_values,
    __global const uint* restrict vector_values,
    __global uint* restrict output_vector_values,
    const unsigned int nrows
)
{
    const unsigned int idx = get_global_id(0);
    if (idx < nrows) {
        unsigned int sum = 0u;
        for (unsigned int j = csr_row_offsets[idx]; j < csr_row_offsets[idx + 1]; ++j) {
            sum += csr_values[j] * vector_values[csr_columns[j]];
        }
        output_vector_values[idx] = sum;
    }
}
