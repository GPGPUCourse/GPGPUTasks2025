#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* cols,
    __global const uint* values,
    __global const uint* vector,
    __global uint* output
) {
    uint row = get_group_id(0);
    uint lid = get_local_id(0);
    uint l = row_offsets[row], r = row_offsets[row + 1];
    __local uint accum[GROUP_SIZE];
    accum[lid] = 0;
    for (uint shift = 0; shift < (r - l); shift += GROUP_SIZE) {
        uint idx = shift + lid;
        if (l + idx < r) {
            uint col = cols[l + idx];
            uint val = values[l + idx];
            accum[lid] += val * vector[col];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // Reduce within workgroup
    for (uint offset = 1; offset < GROUP_SIZE; offset *= 2) {
        if (lid % (2 * offset) == 0) {
            accum[lid] += accum[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        output[row] = accum[0];
    }
}
