#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offset,
    __global const uint* col_ind,
    __global const uint* vals,
    uint row_count,
    __global const uint* vector,
    __global uint* result)
{
    const unsigned int index = get_global_id(0);

    if (index < row_count) {
        int sum = 0;
        for (int i = row_offset[index]; i < row_offset[index + 1]; ++i) {
            sum += vals[i] * vector[col_ind[i]];
        }
        result[index] = sum;
    }
}
