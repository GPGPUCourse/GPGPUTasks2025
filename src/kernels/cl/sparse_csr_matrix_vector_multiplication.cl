#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint *row_offsets,
    __global const uint *columns,
    __global const uint *values,
    __global const uint *vector,
    __global uint *result,
    const uint rows,
    const uint nnz)
{
    __local uint sum[GROUP_SIZE];
    uint row = get_group_id(0);
    uint local_index = get_local_id(0);

    sum[local_index] = 0;

    if (row < rows) {
        uint start = row_offsets[row];
        uint end = nnz;
        if (row + 1 < rows) {
            end = row_offsets[row + 1];
        }

        for (uint i = start + local_index; i < end; i += GROUP_SIZE) {
            sum[local_index] += vector[columns[i]] * values[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_index == 0) {
        uint res = 0;
        for (uint i = 0; i < GROUP_SIZE; ++i) {
            res += sum[i];
        }
        result[row] = res;
    }
}
