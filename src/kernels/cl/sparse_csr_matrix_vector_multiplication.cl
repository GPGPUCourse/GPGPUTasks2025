#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void sparse_csr_matrix_vector_multiplication(
                    __global const uint* row_offsests,
                    __global const uint* columns,
                    __global const uint* values,
                    __global const uint* v,
                    __global uint* output,
                    const uint rows
)
{
    const uint row_id = get_group_id(0);
    const uint local_id = get_local_id(0);
    const uint group_size = get_local_size(0);

    if (row_id >= rows) {
        return;
    }

    const uint row_offset_left = row_offsests[row_id];
    const uint row_offset_right = row_offsests[row_id + 1];

    uint accum = 0;
    for (uint i = row_offset_left + local_id; i < row_offset_right; i += GROUP_SIZE) {
        accum += values[i] * v[columns[i]];
    }

    __local uint local_accum[GROUP_SIZE];
    local_accum[local_id] = accum;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = GROUP_SIZE / 2; i > 0; i /= 2) {
        if (local_id < i) {
            local_accum[local_id] += local_accum[local_id + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        output[row_id] = local_accum[0];
    }
}
