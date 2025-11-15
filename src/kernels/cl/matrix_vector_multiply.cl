#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void matrix_vector_multiply(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output_vector,
    unsigned int nrows
) {
    const unsigned int i = get_global_id(0);
    
    if (i < nrows) {
        unsigned int sum = 0;
        for (unsigned int j = csr_row_offsets[i]; j < csr_row_offsets[i + 1]; j++) {
            sum += csr_values[j] * vector_values[csr_columns[j]];
        }
        
        output_vector[i] = sum;
    }
}
