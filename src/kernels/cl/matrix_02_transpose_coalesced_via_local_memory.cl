#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                       const unsigned int w,
                       const unsigned int h)
{
    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);

    if (row >= h || col >= w) return;

    const unsigned int local_col = get_local_id(0);
    const unsigned int local_row = get_local_id(1);

    __local float local_data[GROUP_SIZE];
    __local float transposed_local_data[GROUP_SIZE];

    local_data[(local_row << 4) + local_col] = matrix[row * w + col];

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int r = (local_row + (local_col & 14)) & 15;

    // построил какую-то +- адекватную биекцию [каждый warp обслуживает что-то типа "диагонали"]
    // например, первый warp обслуживает следующие workItem'ы в workGroup (отмечены звездочками):
    // * * . . . . . . . . . . . . . .
    // * * . . . . . . . . . . . . . .
    // . . * * . . . . . . . . . . . .
    // . . * * . . . . . . . . . . . .
    // . . . . * * . . . . . . . . . .
    // . . . . * * . . . . . . . . . .
    // . . . . . . * * . . . . . . . .
    // . . . . . . * * . . . . . . . .
    // . . . . . . . . * * . . . . . .
    // . . . . . . . . * * . . . . . .
    // . . . . . . . . . . * * . . . .
    // . . . . . . . . . . * * . . . .
    // . . . . . . . . . . . . * * . .
    // . . . . . . . . . . . . * * . .
    // . . . . . . . . . . . . . . * *
    // . . . . . . . . . . . . . . * *

    transposed_local_data[(local_col << 4) + r] = local_data[(r << 4) + local_col];

    barrier(CLK_LOCAL_MEM_FENCE);

    transposed_matrix[col * h + row] = transposed_local_data[(local_col << 4) + local_row];
}
