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
    __global uint* output,
    unsigned int nnz,
    unsigned int nrows,
    unsigned int ncols)
{
    unsigned int group_id = get_group_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);
    
    __local uint buffer[GROUP_SIZE];
    unsigned int lo_offset = csr_row_offsets[group_id];
    unsigned int hi_offset = csr_row_offsets[group_id + 1];

    buffer[local_id] = 0;
    for (unsigned int i = lo_offset + local_id; i < hi_offset; i += GROUP_SIZE) {
        buffer[local_id] += csr_values[i] * vector_values[csr_columns[i]];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    

    if (local_id == 0) {
        unsigned int result = 0;
        for (unsigned int i = 0; i < GROUP_SIZE; i++) {
            result += buffer[i];
        }
        output[group_id] = result;
    }
}
