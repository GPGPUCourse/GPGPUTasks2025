#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* columns,
    __global const uint* csr_values,
    __global const uint* vector_values,
    __global uint* output,
    uint nrows,
    uint ncols)
{
    uint idx = get_global_id(0);
    uint group_id = get_group_id(0);
    uint local_idx = get_local_id(0);

    uint row_start = row_offsets[group_id];
    uint row_end = row_offsets[group_id + 1];

#ifdef _MY_DEBUG
    if (local_idx == 0)
        printf(
            "Group(id %d, [%d, %d))\n",
            group_id, row_start, row_end);
#endif

    __local uint workgroup_data[GROUP_SIZE];

    workgroup_data[local_idx] = 0;
    for (uint i = 0; i + local_idx < (row_end - row_start); i += GROUP_SIZE) {
        workgroup_data[local_idx] += csr_values[i + row_start + local_idx] * vector_values[columns[i + row_start + local_idx]];

#ifdef _MY_DEBUG
        printf(
            "\tKernel(id %d, group %d, local %d) at %d: %d * %d = %d\n",
            idx, group_id, local_idx, i + row_start + local_idx,
            csr_values[i + row_start + local_idx],
            vector_values[columns[i + row_start + local_idx]],
            workgroup_data[local_idx]);
#endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_idx == 0) {
        uint result = 0;
        for (uint i = 0; i < GROUP_SIZE; i++)
            result += workgroup_data[i];
        output[group_id] = result;

#ifdef _MY_DEBUG
        printf("Group(id %d): data = [", group_id);
        for (int i = 0; i < GROUP_SIZE; i++)
            printf("%d ", workgroup_data[i]);
        printf("], result = %d\n", result);
#endif
    }
}
