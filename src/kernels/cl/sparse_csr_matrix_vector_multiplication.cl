#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const unsigned int* csr_row_offsets,
    __global const unsigned int* csr_columns,
    __global const unsigned int* csr_values,
    __global const unsigned int* vector_values,
    __global unsigned int* output_vector_values,
    const unsigned int nonZeroValuesNum,
    const unsigned int numRowOffsets
)
{
    unsigned int groupId = get_group_id(0);
    unsigned int localId = get_local_id(0);
    unsigned int globalId = get_global_id(0);

    __local unsigned int localMem[GROUP_SIZE + 2];

    if (localId == 0) {
        localMem[0] = csr_row_offsets[groupId];
        localMem[1] = csr_row_offsets[groupId + 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int rowStart = localMem[0];
    unsigned int rowEnd = localMem[1];
    unsigned int elemId = localId + rowStart;

    if (elemId < rowEnd) {
        localMem[localId + 2] = csr_values[elemId] * vector_values[csr_columns[elemId]];
    } else {
        localMem[localId + 2] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int acc = 0;

    if (localId == 0) {
        for (unsigned int i = 0; i != (rowEnd - rowStart); ++i) {
            acc += localMem[i + 2];
        }
        output_vector_values[groupId] = acc;
    }
}
