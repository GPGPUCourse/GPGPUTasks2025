#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global uint* offsets,
    __global uint* columns,
    __global uint* values,
    __global uint* vector,
    __global uint* output,
    uint n
) // TODO input/output buffers
{
    uint index = get_global_id(0);
    if (index >= n) {
        return;
    }

    uint sum = 0;
    for (int i = offsets[index]; i < offsets[index + 1]; ++i) {
        sum += values[i] * vector[columns[i]];
    }
    output[index] = sum;
}
