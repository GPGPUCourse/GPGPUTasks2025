#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

// __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector,
    __global uint* output)
{
    const uint groupId = get_group_id(0);
    const uint groupSize = get_local_size(0);

    const uint offset = offsets[groupId];
    const uint nextOffset = offsets[groupId + 1];

    const uint localId = get_local_id(0);
    const uint beg = offset + localId;

    uint sum = 0;
    for (int i = offset + localId; i < nextOffset; i += groupSize) {
        sum += values[i] * vector[columns[i]];
    }
    
    __local uint sums[GROUP_SIZE];
    sums[localId] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int shift = groupSize / 2; shift > 0; shift >>= 1) {
        sums[localId] += ((localId + shift < groupSize) ? sums[localId + shift] : 0);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        output[groupId] = sums[0];
    }
}
