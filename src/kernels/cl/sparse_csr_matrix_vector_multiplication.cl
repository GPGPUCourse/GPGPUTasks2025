#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(uint nrows, uint ncols,
                                                      __global const uint* csr_row_offsets_gpu,
                                                      __global const uint* csr_columns_gpu,
                                                      __global const uint* csr_values_gpu,
                                                      __global const uint* vector_values_gpu,
                                                      __global uint* output_vector_values_gpu)
{
    const uint global_id = get_global_id(0);

    if (global_id >= nrows) {
        return;
    }

    uint row_start = csr_row_offsets_gpu[global_id];
    uint row_end = csr_row_offsets_gpu[global_id + 1];

    uint accumulator = 0u;
    for (uint i = row_start; i < row_end; i++) {
        uint col = csr_columns_gpu[i];
        uint value = csr_values_gpu[i];

        accumulator += value * vector_values_gpu[col];
    }

    output_vector_values_gpu[global_id] = accumulator;
}
