#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector,
    __global uint* output,
    const int nrows)
{
    const uint groupId = get_group_id(0);

    // if (groupId >= nrows) {
    //     printf("i: %d  group: %d  nrows: %d\n",
    //         get_global_id(0), groupId, nrows);
    // }

    const uint offset = offsets[groupId];
    const uint nextOffset = offsets[groupId + 1];

    const uint localId = get_local_id(0);
    const uint beg = offset + localId;

    uint sum = 0;
    for (int i = offset + localId; i < nextOffset; i += GROUP_SIZE) {
        sum += values[i] * vector[columns[i]];
    }
    
    __local uint sums[GROUP_SIZE];
    sums[localId] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int shift = GROUP_SIZE / 2; shift > 0; shift >>= 1) {
        sums[localId] += sums[localId + shift];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        output[groupId] = sums[0];
    }
}
