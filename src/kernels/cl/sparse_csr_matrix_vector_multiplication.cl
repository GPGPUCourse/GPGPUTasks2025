#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_columns,
    __global const uint* csr_row_offsets,
    __global const uint* csr_values,
    __global const uint* input_vector,
    __global       uint* output_vector,
                   int n,
                   int nnz
    )
{
    const uint res_index = get_group_id(0);
    const uint local_idx = get_local_id(0);
    const int first_idx = csr_row_offsets[res_index];
    const int last_idx = csr_row_offsets[res_index + 1];
    __local uint local_arr[GROUP_SIZE];

    local_arr[local_idx] = 0;
    for (int i = first_idx + local_idx; i < last_idx; i += GROUP_SIZE) {
        local_arr[local_idx] += input_vector[csr_columns[i]] * csr_values[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_idx == 0) {
        uint sum = 0;
        for (int i = 0; i < GROUP_SIZE; i++) {
            sum += local_arr[i];
        }
        output_vector[res_index] = sum;
    }
}
