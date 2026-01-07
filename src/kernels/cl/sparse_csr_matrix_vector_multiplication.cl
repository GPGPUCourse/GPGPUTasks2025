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
    unsigned int  nrows,
    unsigned int  ncols,
    unsigned int  nnz
) // TODO input/output buffers
{
    __local uint tile[GROUP_SIZE];
    const unsigned int local_i = get_local_id(0);
    tile[local_i] = 0;

    const unsigned int group_i = get_group_id(0);

    unsigned int l = offsets[group_i];
    unsigned int r = offsets[group_i + 1];
    
    for (unsigned int t = l + local_i; t < r; t += GROUP_SIZE) {
        tile[local_i] += values[t] * vector[columns[t]];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(local_i != 0) return;

    unsigned int sum = 0;
    for(unsigned int t = 0; t < GROUP_SIZE; ++t) {
        sum += tile[t];
    }

    output[group_i] = sum;

}
