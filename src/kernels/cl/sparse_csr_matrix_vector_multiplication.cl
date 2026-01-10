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
    unsigned int  nrows,
    unsigned int  nnz
)
{
    const uint index = get_local_id(0);
    const uint row = get_group_id(0);
    
    __local uint values[GROUP_SIZE];

    if (row < nrows) {
        const uint offset = csr_row_offsets[row];
        const uint csr_index = offset + index;
        const uint offset_next = (row + 1 < nrows) ? csr_row_offsets[row + 1] : nnz;

        if (csr_index < offset_next) {
            const uint col = csr_columns[csr_index];
            values[index] = csr_values[csr_index] * vector_values[col];
        }
        else {
            values[index] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (index > 0 || row >= nrows) return;
    
    uint res = 0;
    for (uint i = 0; i < GROUP_SIZE; i++) {
        res += values[i];
    }
    output_vector_values[row] = res;
}
