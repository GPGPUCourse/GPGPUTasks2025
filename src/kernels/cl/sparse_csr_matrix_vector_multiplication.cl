#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* csr_row_offsets,
    __global const uint* csr_columns,
    __global const uint* csr_values,
    __global const uint* vector,
    __global uint* result,
    const uint nrows,
    const uint ncols)
{
    const uint row = get_global_id(1);
    if (row >= nrows) {
        return;
    }
    const uint local_index = get_local_id(0);

    __local uint accumulator[GROUP_SIZE];

    accumulator[local_index] = 0;

    const uint start = csr_row_offsets[row] + local_index;
    const uint end = csr_row_offsets[row + 1];

    #pragma unroll
    for (uint offset = start; offset < end;
        accumulator[local_index] += csr_values[offset] * vector[csr_columns[offset]],
              offset += GROUP_SIZE) {
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        uint sum = 0;
        #pragma unroll
        for (uint i = 0; i < GROUP_SIZE; sum += accumulator[i++]) {
        }
        result[row] = sum;
    }
}
