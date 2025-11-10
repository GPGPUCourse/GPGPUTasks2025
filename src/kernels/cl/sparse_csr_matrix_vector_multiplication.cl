#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const unsigned int* row_offsets,
    __global const unsigned int* columns,
    __global const unsigned int* matrix_values,
    __global const unsigned int* vector_values,
    __global unsigned int* vector_output,
    unsigned int n_rows,
    unsigned int n_cols,
    unsigned int n_nz)
{
    const unsigned int group_id = get_group_id(0);
    const unsigned int thread_id = get_local_id(0);

    __local unsigned int acc[GROUP_SIZE];

    acc[thread_id] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int row_start = row_offsets[group_id];
    const unsigned int row_end = (group_id + 1 < n_rows) ? row_offsets[group_id + 1] : n_nz;
    for (unsigned int i = row_start + thread_id; i < row_end; i += GROUP_SIZE) {
        acc[thread_id] += matrix_values[i] * vector_values[columns[i]];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (thread_id == 0) {
        unsigned int master_acc = 0;
        for (unsigned int i = 0; i < GROUP_SIZE; i++) {
            master_acc += acc[i];
        }
        vector_output[group_id] = master_acc;
    }
}
