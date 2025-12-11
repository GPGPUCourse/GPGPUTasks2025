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
    __global uint* output,
    uint nrows)
{
    const uint groupId = get_group_id(0);
    const uint groupSize = get_local_size(0);
    uint sum = 0;
    const uint localId = get_local_id(0);

    if (groupId < nrows) {
        const uint offset = offsets[groupId];
        const uint nextOffset = offsets[groupId + 1];

        for (int i = offset + localId; i < nextOffset; i += groupSize) {
            const uint column = columns[i];
            const uint val = values[i];
            const uint vecVal = vector[column];
            sum += val * vecVal;
        }
    }
    
    __local uint sums[GROUP_SIZE];
    sums[localId] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int shift = 1; shift < groupSize; shift <<= 1) {
        uint tmp = ((localId + shift < groupSize) ? sums[localId + shift] : 0);
        barrier(CLK_LOCAL_MEM_FENCE);
        sums[localId] += tmp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0 && groupId < nrows) {
        output[groupId] = sums[0];
    }
}
