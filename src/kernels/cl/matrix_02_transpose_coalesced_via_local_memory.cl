#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

// require that the WorkGroup size is sqare or X axis is less than Y axis
#if (GROUP_SIZE_X > GROUP_SIZE_Y)
#error GROUP_SIZE_X must be less than or equal to GROUP_SIZE_Y
#endif

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float* matrix, // w x h
    __global float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{

    const unsigned int global_idx_x = get_global_id(0);
    const unsigned int global_idx_y = get_global_id(1);

    const unsigned int local_idx_x = get_local_id(0);
    const unsigned int local_idx_y = get_local_id(1);

    const unsigned int SHIFTED_GROUP_SIZE_X = GROUP_SIZE_X + 1;

    __local float local_matrix[SHIFTED_GROUP_SIZE_X * GROUP_SIZE_Y];

    const bool in_bounds_in = global_idx_x < w && global_idx_y < h;
    // coalesced read
    local_matrix[local_idx_y * SHIFTED_GROUP_SIZE_X + local_idx_x] = in_bounds_in ? matrix[global_idx_y * w + global_idx_x] : 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int transposed_idx_x = global_idx_y - local_idx_y + local_idx_x;
    const unsigned int transposed_idx_y = global_idx_x - local_idx_x + local_idx_y;

    const bool in_bounds_out = transposed_idx_x < h && transposed_idx_y < w;
    if (in_bounds_out) {
        // coalesced write
        transposed_matrix[transposed_idx_y * h + transposed_idx_x] = local_matrix[local_idx_x * SHIFTED_GROUP_SIZE_X + local_idx_y];
    }
}