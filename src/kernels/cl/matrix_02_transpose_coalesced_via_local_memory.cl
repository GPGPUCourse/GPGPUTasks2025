#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(32, 32, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int x_loc = get_local_id(0);
    const unsigned int y_loc = get_local_id(1);

    __local float local_data[32][33];

    if (x < w && y < h) {
        local_data[y_loc][x_loc] = matrix[y * w + x];
    } else {
        local_data[y_loc][x_loc] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < h && y < w) {
        transposed_matrix[(y - y_loc) + x_loc + ((x - x_loc) + y_loc) * h] = local_data[x_loc][y_loc];
    }
}
