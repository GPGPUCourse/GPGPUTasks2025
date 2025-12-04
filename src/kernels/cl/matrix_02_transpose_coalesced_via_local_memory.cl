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
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint x_loc = get_local_id(0);
    const uint y_loc = get_local_id(1);

    __local float local_data[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    if (x < w && y < h) {
        local_data[y_loc][x_loc] = matrix[y * w + x];
    } else {
        local_data[y_loc][x_loc] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint y_T = x - x_loc + y_loc;
    const uint x_T = y - y_loc + x_loc;

    if (y_T < w && x_T < h) {
        transposed_matrix[y_T * h + x_T] = local_data[x_loc][y_loc];
    }
}
