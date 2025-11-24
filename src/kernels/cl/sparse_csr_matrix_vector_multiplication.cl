#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector,
    __global uint* output,
    uint rows,
    uint col_size
)
{
    uint i = get_global_id(1);
    uint j = get_global_id(0);

    uint nz_count = offsets[i + 1] - offsets[i];
    uint read_from = offsets[i] + j;

    uint sum = 0;
    while (read_from < offsets[i] + nz_count) {
        sum += values[read_from] * vector[columns[read_from]];
        read_from += GROUP_SIZE;
    }

    atomic_add(&output[i], sum);
}
