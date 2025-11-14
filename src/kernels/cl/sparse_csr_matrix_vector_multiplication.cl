#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* row_offsets,
    __global const uint* values,
    __global const uint* columns,
    __global const uint* vector_values,
    __global uint* out_vector,
    uint nnz_cnt,
    uint rows_cnt,
    uint cols_cnt)
{
    const uint group_id = get_global_id(0) / GROUP_SIZE;
    const uint local_id = get_local_id(0);
    if (group_id >= rows_cnt) {
        return;
    }
    __local uint sum;
    if (local_id == 0) {
        sum = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint from = row_offsets[group_id];
    const uint to = row_offsets[group_id + 1];
    uint iters = (to - from) / GROUP_SIZE + 1;
    uint sum_loc = 0;
    for (uint i = 0; i < iters; ++i) {
        const uint elem_idx = from + i * GROUP_SIZE + local_id;
        if (elem_idx >= to) {
            break;
        }
        const uint column = columns[elem_idx];
        sum_loc += values[elem_idx] * vector_values[column];
    }
    atomic_add(&sum, sum_loc);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        out_vector[group_id] = sum;
    }
}
