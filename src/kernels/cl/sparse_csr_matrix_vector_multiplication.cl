#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const unsigned int* restrict csr_row_offsets,
    __global const unsigned int* restrict csr_columns,
    __global const unsigned int* restrict csr_values,
    __global const unsigned int* restrict vector_values,
    __global       unsigned int* restrict result_vector_values,
                   unsigned int  nrows
)
{
    const uint globalX = get_global_id(0);
    const uint globalY = get_global_id(1);
    const uint localY = get_local_id(1);

    __local uint accs[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    uint count = 0;
    uint acc = 0;
    bool process = globalY < nrows;
    if (process) {
        const uint2 offsets = vload2(0, &csr_row_offsets[globalY]);
        uint beginOffset = offsets.x;
        uint endOffset = offsets.y;
        count = endOffset - beginOffset;
        for (uint i = beginOffset + globalX; i < endOffset; i += GROUP_SIZE_X) {
            acc += csr_values[i] * vector_values[csr_columns[i]];
        }
        accs[localY][globalX] = acc;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (globalX == 0 && process) {
        for (uint i = 1; i < min((uint)GROUP_SIZE_X, count); ++i) {
            acc += accs[localY][i];
        }
        result_vector_values[globalY] = acc;
    }
}
