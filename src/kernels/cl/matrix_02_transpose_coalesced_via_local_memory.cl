#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float local_mem[GROUP_SIZE_X][GROUP_SIZE_Y];

    int x_index = get_global_id(0);
    int y_index = get_global_id(1);

    int x_local = get_local_id(0);
    int y_local = get_local_id(1);

    if (x_index < w && y_index < h) {
        int index_in = (y_index) * w + x_index;
        local_mem[y_local][x_local] = matrix[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int x_transposed = y_index + x_local - y_local;
    int y_transposed = x_index + y_local - x_local;

    if (x_transposed < h && y_transposed < w) {
        int index_out = y_transposed * h + x_transposed;
        transposed_matrix[index_out] = local_mem[x_local][y_local];
    }

}
